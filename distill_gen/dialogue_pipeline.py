"""
对话模式主流程编排模块。

串联整个对话蒸馏生成 pipeline:
  1. 加载配置
  2. 扫描 JSON 数据
  3. Checkpoint 恢复（断点续传）
  4. 批量并发生成（单次 LLM 调用返回 JSON 数组）
  5. 变换为 OpenAI SFT 格式（messages 数组 + reasoning_content）
  6. 按 source_file 分组写入 JSON
  7. 生成统计报告
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from distill_gen.config import Config, load_config, resolve_path
from distill_gen.dialogue_generator import DialogueGeneratedItem, DialogueGenerator
from distill_gen.loader import DataItem, DataLoader
from distill_gen.pipeline import Checkpoint, setup_logging
from distill_gen.writer import JsonWriter

logger = logging.getLogger(__name__)


def _to_output_format(item: DialogueGeneratedItem) -> dict:
    """
    将 DialogueGeneratedItem 转为 OpenAI SFT 格式。

    输出格式:
    {
        "id": ...,
        "type": ...,
        "difficulty": ...,
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "reasoning_content": "...", "content": "..."},
            ...
        ]
    }
    """
    messages = [
        {"role": "system", "content": item.system},
    ]

    for r in item.rounds:
        messages.append({"role": "user", "content": r.instruction})
        messages.append({
            "role": "assistant",
            "reasoning_content": r.thinking,
            "content": r.output,
        })

    return {
        "id": item.data_item.id,
        "type": item.data_item.type,
        "difficulty": item.data_item.difficulty,
        "messages": messages,
    }


class DialoguePipeline:
    """对话模式主流程。"""

    def __init__(self, config: Config):
        self.config = config
        self.logger = logger

        self.input_dir = resolve_path(config.data.input_dir)
        self.output_dir = resolve_path(config.data.output_dir)
        self.dialogue_output_dir = self.output_dir / "dialogue"
        self.checkpoint_dir = resolve_path(config.data.checkpoint_dir)

        self.loader = DataLoader(config)
        self.writer = JsonWriter(str(self.dialogue_output_dir))
        self.checkpoint = Checkpoint(self.checkpoint_dir)

        self._source_results: dict[str, list[dict]] = {}

        self.stats = {
            "total": 0,
            "pending": 0,
            "generated": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "total_rounds": 0,
        }

    async def run(
        self,
        limit: Optional[int] = None,
        num_rounds: int = 15,
    ):
        gen_start = datetime.now()
        self.logger.info("=" * 60)
        self.logger.info("对话蒸馏数据生成系统启动")
        self.logger.info(f"输入目录: {self.input_dir}")
        self.logger.info(f"输出目录: {self.dialogue_output_dir}")
        self.logger.info(f"目标轮次: {num_rounds} 轮/条")
        self.logger.info(f"Checkpoint 目录: {self.checkpoint_dir}")
        self.logger.info("=" * 60)

        all_items = self.loader.load_all_json_files()
        self.stats["total"] = len(all_items)

        if not all_items:
            self.logger.warning("没有找到有效数据，退出")
            return

        pending_items = [
            item for item in all_items
            if not self.checkpoint.is_completed(item.source_checksum)
        ]
        self.stats["pending"] = len(pending_items)
        self.stats["skipped"] = len(all_items) - len(pending_items)

        self.logger.info(
            f"总计 {len(all_items)} 条 | 已完成 {self.stats['skipped']} | "
            f"待处理 {len(pending_items)}"
        )

        if not pending_items:
            self.logger.info("所有条目已完成")
            self._rebuild_from_checkpoint(all_items, gen_start)
            return

        if limit and limit > 0:
            pending_items = pending_items[:limit]
            self.logger.info(f"测试模式：仅处理前 {limit} 条")

        generator = DialogueGenerator(self.config)

        try:
            self.logger.info("开始批量对话生成...")
            all_results: list[DialogueGeneratedItem] = []

            batch_size = self.config.concurrency.max_workers
            for i in range(0, len(pending_items), batch_size):
                batch = pending_items[i : i + batch_size]
                batch_num = i // batch_size + 1
                total_batches = (len(pending_items) + batch_size - 1) // batch_size

                self.logger.info(
                    f"处理批次 {batch_num}/{total_batches} ({len(batch)} 条)"
                )

                batch_start = datetime.now()
                results = await generator.generate_batch(batch, num_rounds)
                batch_elapsed = (datetime.now() - batch_start).total_seconds()

                for result in results:
                    result.system = self.config.system_mapping.get_system(
                        result.data_item.type
                    )
                    result.system_chars = len(result.system)

                    if result.passed:
                        self.checkpoint.mark_completed(
                            result.data_item.source_checksum
                        )

                    if result.passed:
                        self.stats["passed"] += 1
                    else:
                        self.stats["failed"] += 1
                    self.stats["generated"] += 1
                    self.stats["total_rounds"] += result.num_rounds

                    all_results.append(result)

                    output = _to_output_format(result)
                    source = result.data_item.source_file
                    self._source_results.setdefault(source, []).append(output)
                    self.writer.write_dialogue_json_file(
                        source, self._source_results[source]
                    )

                    status = "✅" if result.passed else "❌"
                    self.logger.info(
                        f"  {status} [{result.data_item.source_file}"
                        f"#{result.data_item.id}] "
                        f"{result.data_item.type}/"
                        f"{result.data_item.difficulty} | "
                        f"{result.num_rounds}轮 | "
                        f"耗时 {result.generation_time:.1f}s | "
                        f"thinking={result.total_thinking_chars}字 "
                        f"output={result.total_output_chars}字"
                    )

                self.checkpoint.save()

                done = self.stats["skipped"] + self.stats["generated"]
                pct = done / self.stats["total"] * 100 if self.stats["total"] > 0 else 0
                self.logger.info(
                    f"批次 {batch_num}/{total_batches} 完成 | "
                    f"耗时 {batch_elapsed:.1f}s | "
                    f"进度 {done}/{self.stats['total']} ({pct:.1f}%) | "
                    f"通过 {self.stats['passed']} | 失败 {self.stats['failed']}"
                )

        finally:
            await generator.close()

        self._finalize(all_results, gen_start)

    def _finalize(self, all_results: list[DialogueGeneratedItem], gen_start: datetime):
        gen_end = datetime.now()
        self.writer.write_dialogue_stats(all_results, gen_start, gen_end)

        total = len(all_results)
        passed = sum(1 for r in all_results if r.passed)
        failed = total - passed
        total_rounds = sum(r.num_rounds for r in all_results)
        elapsed = (gen_end - gen_start).total_seconds()

        self.logger.info("=" * 60)
        self.logger.info("对话生成完成！汇总：")
        self.logger.info(f"  总条目数: {total}")
        self.logger.info(f"  总轮次数: {total_rounds}")
        pct = passed / total * 100 if total else 0
        self.logger.info(f"  通过: {passed} ({pct:.1f}%)")
        pct_f = failed / total * 100 if total else 0
        self.logger.info(f"  失败: {failed} ({pct_f:.1f}%)")
        self.logger.info(f"  总耗时: {elapsed:.1f}s ({elapsed / 60:.1f}min)")
        if total > 0:
            self.logger.info(f"  平均耗时: {elapsed / total:.1f}s/条")
        self.logger.info(f"  输出目录: {self.dialogue_output_dir}")
        self.logger.info(f"  统计报告: {self.dialogue_output_dir / '_dialogue_stats.md'}")
        self.logger.info("=" * 60)

    def _rebuild_from_checkpoint(
        self, all_items: list[DataItem], gen_start: datetime
    ):
        self.logger.info("所有条目已在 checkpoint 中标记完成，直接统计已有结果...")
        all_results: list[DialogueGeneratedItem] = []
        for item in all_items:
            if self.checkpoint.is_completed(item.source_checksum):
                all_results.append(DialogueGeneratedItem(
                    data_item=item,
                    passed=True,
                ))
        if all_results:
            self._finalize(all_results, gen_start)
        else:
            self.logger.warning("checkpoint 中无已完成条目")


def parse_args():
    parser = argparse.ArgumentParser(
        description="对话式蒸馏数据生成 — 生成多轮对话 JSON（OpenAI SFT 格式）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m distill_gen.dialogue_pipeline                                   # 默认 15 轮
  python -m distill_gen.dialogue_pipeline --config config_gen.yaml --num-rounds 10
  python -m distill_gen.dialogue_pipeline --limit 5 --verbose
        """,
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config_gen.yaml",
        help="YAML 配置文件路径 (默认: config_gen.yaml)",
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=None,
        help="限制处理的条目数（用于测试验证）",
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=15,
        help="每条目生成的对话轮次 (默认: 15)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="启用 DEBUG 级别日志",
    )
    return parser.parse_args()


async def main():
    args = parse_args()
    setup_logging(args.verbose)

    config = load_config(args.config)

    config_path = Path(args.config)
    if config_path.exists() and config_path.parent != Path("."):
        base_dir = config_path.parent.resolve()
        config.data.input_dir = str(resolve_path(config.data.input_dir, base_dir))

    logger.info(f"配置文件: {args.config}")
    logger.info(f"llama.cpp 服务: {config.llama_cpp.base_url}")
    logger.info(f"模型: {config.llama_cpp.model}")
    logger.info(f"对话轮次: {args.num_rounds} 轮/条")

    pipeline = DialoguePipeline(config)
    await pipeline.run(limit=args.limit, num_rounds=args.num_rounds)


if __name__ == "__main__":
    asyncio.run(main())
