"""
主流程编排模块。

串联整个蒸馏数据生成 pipeline:
  1. 加载配置
  2. 扫描 JSON 数据
  3. Checkpoint 恢复（断点续传）
  4. 批量并发生成
  5. 按 source_file 分组写入 JSON（字段与原始文件一致）
  6. 生成统计报告
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
from distill_gen.generator import GeneratedItem, Generator
from distill_gen.loader import DataItem, DataLoader
from distill_gen.writer import JsonWriter

# ============================================================
# 日志配置
# ============================================================

def setup_logging(verbose: bool = False):
    """配置日志系统。"""
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d — %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        handlers=[logging.StreamHandler(sys.stderr)],
    )


# ============================================================
# Checkpoint 管理
# ============================================================

class Checkpoint:
    """
    简易断点续传管理器。

    记录已成功生成的条目 checksum，下次启动自动跳过。
    """

    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / "completed.json"
        self._completed: set[str] = set()
        self._load()

    @property
    def completed(self) -> set[str]:
        return self._completed

    def is_completed(self, checksum: str) -> bool:
        return checksum in self._completed

    def mark_completed(self, checksum: str):
        self._completed.add(checksum)

    def save(self):
        """持久化已完成的 checksum 列表。"""
        data = {
            "count": len(self._completed),
            "checksums": sorted(self._completed),
        }
        self.checkpoint_file.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load(self):
        """从文件恢复已完成列表。"""
        if self.checkpoint_file.exists():
            try:
                data = json.loads(self.checkpoint_file.read_text(encoding="utf-8"))
                self._completed = set(data.get("checksums", []))
                logging.getLogger(__name__).info(
                    f"Checkpoint 已恢复: {len(self._completed)} 条已完成"
                )
            except Exception as e:
                logging.getLogger(__name__).warning(f"Checkpoint 读取失败: {e}")
                self._completed = set()


# ============================================================
# 主流程
# ============================================================

class Pipeline:
    """
    蒸馏数据生成主流程。
    """

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # 解析路径
        self.input_dir = resolve_path(config.data.input_dir)
        self.output_dir = resolve_path(config.data.output_dir)
        self.checkpoint_dir = resolve_path(config.data.checkpoint_dir)

        # 初始化模块
        self.loader = DataLoader(config)
        self.writer = JsonWriter(str(self.output_dir))
        self.checkpoint = Checkpoint(self.checkpoint_dir)

        # 统计
        self.stats = {
            "total": 0,
            "pending": 0,
            "generated": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
        }

    async def run(self, limit: Optional[int] = None):
        """
        执行完整 pipeline。

        Args:
            limit: 限制处理的条目数（用于测试），None 表示全量
        """
        gen_start = datetime.now()
        self.logger.info("=" * 60)
        self.logger.info("蒸馏数据生成系统启动")
        self.logger.info(f"输入目录: {self.input_dir}")
        self.logger.info(f"输出目录: {self.output_dir}")
        self.logger.info(f"Checkpoint 目录: {self.checkpoint_dir}")
        self.logger.info("=" * 60)

        # Step 1: 加载数据
        all_items = self.loader.load_all_json_files()
        self.stats["total"] = len(all_items)

        if not all_items:
            self.logger.warning("没有找到有效数据，退出")
            return

        # Step 2: 过滤已完成的条目
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
            self.logger.info("所有条目已完成，无需生成")
            self._rebuild_json_from_checkpoint(all_items, gen_start)
            return

        # 限制处理数量
        if limit and limit > 0:
            pending_items = pending_items[:limit]
            self.logger.info(f"测试模式：仅处理前 {limit} 条")

        # Step 3: 初始化生成器
        generator = Generator(self.config)

        try:
            # Step 4: 分批并发生成
            self.logger.info("开始批量生成...")
            all_results: list[GeneratedItem] = []

            batch_size = self.config.concurrency.max_workers
            for i in range(0, len(pending_items), batch_size):
                batch = pending_items[i:i + batch_size]
                batch_num = i // batch_size + 1
                total_batches = (len(pending_items) + batch_size - 1) // batch_size

                self.logger.info(
                    f"处理批次 {batch_num}/{total_batches} "
                    f"({len(batch)} 条)"
                )

                results = await generator.generate_batch(batch)

                for result in results:
                    # 从配置映射表注入 system 字段
                    result.system = self.config.system_mapping.get_system(
                        result.data_item.type
                    )
                    result.system_chars = len(result.system)

                    # 更新 checkpoint（仅标记通过的）
                    if result.passed:
                        self.checkpoint.mark_completed(result.data_item.source_checksum)

                    # 更新统计
                    if result.passed:
                        self.stats["passed"] += 1
                    else:
                        self.stats["failed"] += 1
                    self.stats["generated"] += 1

                    all_results.append(result)

                # 每批次后保存 checkpoint
                self.checkpoint.save()

                # 进度报告
                done = self.stats["skipped"] + self.stats["generated"]
                pct = done / self.stats["total"] * 100 if self.stats["total"] > 0 else 0
                self.logger.info(
                    f"进度: {done}/{self.stats['total']} ({pct:.1f}%) | "
                    f"通过: {self.stats['passed']} | "
                    f"失败: {self.stats['failed']}"
                )

        finally:
            await generator.close()

        # Step 5: 写入 JSON + 统计
        self._finalize(all_results, gen_start)

    def _finalize(self, all_results: list[GeneratedItem], gen_start: datetime):
        """写入 JSON 文件和统计报告。"""
        gen_end = datetime.now()

        # 按 source_file 分组写入 JSON
        self.logger.info("写入 JSON 文件...")
        self.writer.write_all(all_results)

        # 写入统计
        self.writer.write_stats(all_results, gen_start, gen_end)

        # 打印汇总
        total = len(all_results)
        passed = sum(1 for r in all_results if r.passed)
        failed = total - passed
        elapsed = (gen_end - gen_start).total_seconds()

        self.logger.info("=" * 60)
        self.logger.info("生成完成！汇总：")
        self.logger.info(f"  总条目数: {total}")
        pct = passed / total * 100 if total else 0
        self.logger.info(f"  通过: {passed} ({pct:.1f}%)")
        pct_f = failed / total * 100 if total else 0
        self.logger.info(f"  失败: {failed} ({pct_f:.1f}%)")
        self.logger.info(f"  总耗时: {elapsed:.1f}s ({elapsed / 60:.1f}min)")
        if total > 0:
            self.logger.info(f"  平均耗时: {elapsed / total:.1f}s/条")
        self.logger.info(f"  输出目录: {self.output_dir}")
        self.logger.info(f"  统计报告: {self.output_dir / '_stats.md'}")
        self.logger.info("=" * 60)

    def _rebuild_json_from_checkpoint(
        self, all_items: list[DataItem], gen_start: datetime
    ):
        """
        当所有条目已完成时，从已有输出 JSON 中重建结果并输出统计。
        适用于断点续传后全部完成的场景。
        """
        self.logger.info("所有条目已在 checkpoint 中标记完成，直接统计已有结果...")

        # 从已输出的 JSON 文件中加载结果来构建统计
        all_results: list[GeneratedItem] = []
        for item in all_items:
            if self.checkpoint.is_completed(item.source_checksum):
                all_results.append(GeneratedItem(
                    data_item=item,
                    passed=True,
                ))

        if all_results:
            self._finalize(all_results, gen_start)
        else:
            self.logger.warning("checkpoint 中无已完成条目，请检查数据一致性")


# ============================================================
# 命令行入口
# ============================================================

def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="蒸馏数据生成系统 — 使用 llama.cpp 生成高质量 thinking + output（JSON 格式输出）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m distill_gen.pipeline                           # 使用默认配置
  python -m distill_gen.pipeline --config config_gen.yaml  # 指定配置文件
  python -m distill_gen.pipeline --limit 10                # 测试模式（仅处理 10 条）
  python -m distill_gen.pipeline --verbose                 # 详细日志
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
        "--verbose", "-v",
        action="store_true",
        help="启用 DEBUG 级别日志",
    )
    return parser.parse_args()


async def main():
    """主函数入口。"""
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # 加载配置
    config = load_config(args.config)

    # 用命令行参数覆盖数据目录（确保使用配置文件所在目录）
    config_path = Path(args.config)
    if config_path.exists() and config_path.parent != Path("."):
        base_dir = config_path.parent.resolve()
        config.data.input_dir = str(resolve_path(config.data.input_dir, base_dir))

    logger.info(f"配置文件: {args.config}")
    logger.info(f"llama.cpp 服务: {config.llama_cpp.base_url}")
    logger.info(f"模型: {config.llama_cpp.model}")

    # 运行 pipeline
    pipeline = Pipeline(config)
    await pipeline.run(limit=args.limit)


if __name__ == "__main__":
    asyncio.run(main())
