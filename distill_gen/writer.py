"""
JSON 输出模块。

负责:
- 将 GeneratedItem 按 source_file 分组，输出为 JSON 文件
- JSON 字段: id, type, difficulty, system, instruction, input, thinking, output
- 目录结构: gen_output/{source_filename}（与原始文件名一致）
- 生成统计 _stats.md（Markdown 格式）
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from distill_gen.generator import GeneratedItem

logger = logging.getLogger(__name__)


class JsonWriter:
    """
    JSON 格式输出器。

    输出 JSON 字段:
        id, type, difficulty, system, instruction, input, thinking, output

    目录结构:
        {output_dir}/
        ├── deeplearn_llm_0_100.json      # 与原始文件同名
        ├── gpt-vllm_0_10.json
        ├── ...
        └── _stats.md                      # 生成统计（Markdown）
    """

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_json_file(
        self,
        source_file: str,
        items: list[GeneratedItem],
    ) -> Path:
        """
        将一个源文件对应的所有生成结果写入单个 JSON 文件。

        Args:
            source_file: 原始 JSON 文件名
            items: 该源文件对应的所有生成结果

        Returns:
            Path: 写入的 JSON 文件路径
        """
        output_records = []
        for item in items:
            di = item.data_item
            record = {
                "id": di.id,
                "type": di.type,
                "difficulty": di.difficulty,
                "system": item.system,
                "instruction": di.instruction,
                "input": di.input,
                "thinking": item.thinking,
                "output": item.output,
            }
            output_records.append(record)

        # 按原始 id 排序
        output_records.sort(key=lambda r: r["id"])

        filepath = self.output_dir / source_file
        filepath.write_text(
            json.dumps(output_records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        logger.info(
            f"写入 JSON: {source_file} ({len(output_records)} 条)"
        )
        return filepath

    def write_all(
        self,
        all_items: list[GeneratedItem],
    ) -> list[Path]:
        """
        将所有生成结果按 source_file 分组写入对应的 JSON 文件。

        Args:
            all_items: 所有生成结果

        Returns:
            list[Path]: 写入的所有文件路径
        """
        by_source: dict[str, list[GeneratedItem]] = {}
        for item in all_items:
            source = item.data_item.source_file
            by_source.setdefault(source, []).append(item)

        written_paths = []
        for source_file, items in sorted(by_source.items()):
            filepath = self.write_json_file(source_file, items)
            written_paths.append(filepath)

        return written_paths

    def write_stats(
        self,
        all_items: list[GeneratedItem],
        gen_start_time: Optional[datetime] = None,
        gen_end_time: Optional[datetime] = None,
    ):
        """
        写入生成统计 _stats.md（Markdown 格式）。

        Args:
            all_items: 所有生成结果
            gen_start_time: 生成开始时间
            gen_end_time: 生成结束时间
        """
        total = len(all_items)
        passed = sum(1 for i in all_items if i.passed)
        failed = total - passed
        pass_rate = passed / total * 100 if total > 0 else 0
        fail_rate = failed / total * 100 if total > 0 else 0

        # 各维度平均统计
        total_system_chars = sum(i.system_chars for i in all_items)
        total_thinking = sum(i.thinking_chars for i in all_items)
        total_output = sum(i.output_chars for i in all_items)
        total_time = sum(i.generation_time for i in all_items)

        avg_system = total_system_chars / total if total > 0 else 0
        avg_thinking = total_thinking / total if total > 0 else 0
        avg_output = total_output / total if total > 0 else 0
        avg_time = total_time / total if total > 0 else 0

        # 耗时分布（min / median / P95 / max）
        gen_times = sorted(i.generation_time for i in all_items if i.generation_time > 0)
        time_min = gen_times[0] if gen_times else 0
        time_max = gen_times[-1] if gen_times else 0
        time_median = gen_times[len(gen_times) // 2] if gen_times else 0
        time_p95 = gen_times[int(len(gen_times) * 0.95)] if len(gen_times) >= 20 else (time_max if gen_times else 0)

        # 按难度统计
        by_difficulty: dict[str, dict] = {}
        for item in all_items:
            diff = item.data_item.difficulty
            if diff not in by_difficulty:
                by_difficulty[diff] = {"total": 0, "passed": 0, "failed": 0}
            by_difficulty[diff]["total"] += 1
            if item.passed:
                by_difficulty[diff]["passed"] += 1
            else:
                by_difficulty[diff]["failed"] += 1

        # 按 type 统计（按 total 降序）
        by_type: dict[str, dict] = {}
        for item in all_items:
            t = item.data_item.type
            if t not in by_type:
                by_type[t] = {"total": 0, "passed": 0, "failed": 0}
            by_type[t]["total"] += 1
            if item.passed:
                by_type[t]["passed"] += 1
            else:
                by_type[t]["failed"] += 1

        sorted_types = sorted(by_type.items(), key=lambda x: -x[1]["total"])

        # 构建 Markdown 统计报告
        lines = [
            "# 蒸馏数据生成统计报告",
            "",
            "## 基本信息",
            "",
            f"| 项目 | 值 |",
            f"|------|-----|",
            f"| 生成开始时间 | {gen_start_time.strftime('%Y-%m-%d %H:%M:%S') if gen_start_time else 'N/A'} |",
            f"| 生成结束时间 | {gen_end_time.strftime('%Y-%m-%d %H:%M:%S') if gen_end_time else 'N/A'} |",
            f"| 总耗时 | {self._fmt_duration(gen_start_time, gen_end_time)} |",
            f"| 输出目录 | `{self.output_dir}` |",
            "",
            "## 生成概览",
            "",
            f"| 指标 | 值 |",
            f"|------|-----|",
            f"| 总条目数 | {total} |",
            f"| 通过数 | {passed} ({pass_rate:.1f}%) |",
            f"| 失败数 | {failed} ({fail_rate:.1f}%) |",
            "",
            "## 字数统计（平均）",
            "",
            f"| 字段 | 平均字数 |",
            f"|------|---------|",
            f"| system (角色描述) | {avg_system:.0f} 字 |",
            f"| thinking (思维链) | {avg_thinking:.0f} 字 |",
            f"| output (最终答案) | {avg_output:.0f} 字 |",
            f"| 单条平均耗时 | {avg_time:.1f}s |",
            f"| 总生成耗时 | {total_time:.0f}s ({total_time / 60:.1f}min) |",
            "",
            "## 耗时分布",
            "",
            f"| 指标 | 值 |",
            f"|------|-----|",
            f"| 最快 | {time_min:.1f}s |",
            f"| 中位数 | {time_median:.1f}s |",
            f"| P95 | {time_p95:.1f}s |",
            f"| 最慢 | {time_max:.1f}s |",
            f"| 平均 | {avg_time:.1f}s |",
            "",
            "## 按难度统计",
            "",
            "| 难度 | 总数 | 通过 | 失败 | 通过率 |",
            "|------|------|------|------|--------|",
        ]

        for diff in ["初级", "中级", "高级", "专家"]:
            if diff in by_difficulty:
                d = by_difficulty[diff]
                rate = d["passed"] / d["total"] * 100 if d["total"] > 0 else 0
                lines.append(
                    f"| {diff} | {d['total']} | {d['passed']} | "
                    f"{d['failed']} | {rate:.1f}% |"
                )

        lines.extend([
            "",
            "## 按技术领域统计",
            "",
            "| 领域 | 总数 | 通过 | 失败 | 通过率 |",
            "|------|------|------|------|--------|",
        ])

        for t_name, t_data in sorted_types:
            rate = t_data["passed"] / t_data["total"] * 100 if t_data["total"] > 0 else 0
            lines.append(
                f"| {t_name} | {t_data['total']} | {t_data['passed']} | "
                f"{t_data['failed']} | {rate:.1f}% |"
            )

        lines.extend([
            "",
            "## 源文件明细",
            "",
            "| 源文件 | 条目数 | 通过 | 失败 |",
            "|--------|--------|------|------|",
        ])

        by_source: dict[str, list[GeneratedItem]] = {}
        for item in all_items:
            source = item.data_item.source_file
            by_source.setdefault(source, []).append(item)

        for source_file, items in sorted(by_source.items()):
            p = sum(1 for i in items if i.passed)
            f = len(items) - p
            lines.append(f"| {source_file} | {len(items)} | {p} | {f} |")

        lines.extend([
            "",
            f"---",
            f"*报告生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        ])

        stats_path = self.output_dir / "_stats.md"
        stats_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"统计报告已写入: {stats_path}")

    @staticmethod
    def _fmt_duration(
        start: Optional[datetime], end: Optional[datetime]
    ) -> str:
        """格式化耗时。"""
        if start is None or end is None:
            return "N/A"
        secs = (end - start).total_seconds()
        if secs < 60:
            return f"{secs:.0f}s"
        mins = secs / 60
        if mins < 60:
            return f"{mins:.1f}min"
        hours = mins / 60
        return f"{hours:.1f}h ({mins:.0f}min)"
