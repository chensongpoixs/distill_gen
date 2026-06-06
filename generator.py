"""
核心生成引擎模块。

负责:
- Prompt 构建（难度分层，仅要求 thinking + output）
- 调用 LLMClient 进行文本生成
- 解析 LLM 响应（提取 thinking / output，system 由配置映射表提供）
- 长度校验与重试
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass

from distill_gen.config import Config
from distill_gen.llm_client import LLMClient
from distill_gen.loader import DataItem

logger = logging.getLogger(__name__)


@dataclass
class GeneratedItem:
    """生成结果（system 由配置表映射，不从 LLM 生成）"""
    data_item: DataItem
    system: str = ""
    thinking: str = ""
    output: str = ""
    system_chars: int = 0
    thinking_chars: int = 0
    output_chars: int = 0
    retry_count: int = 0
    generation_time: float = 0.0
    passed: bool = True
    error_message: str = ""


# 难度分层 Prompt 后缀
DIFFICULTY_SUFFIX = {
    "初级": (
        "\n\n请用通俗易懂的语言解释基础概念，配合简单示例帮助理解。"
        "思考过程请展示从问题到答案的完整推理链。"
    ),
    "中级": (
        "\n\n请进行对比分析，说明技术选型的工程考量，并提供代码/伪代码示例。"
        "思考过程请展示多方案权衡的推理过程。"
    ),
    "高级": (
        "\n\n请深入分析架构设计决策，讨论性能优化方案，"
        "延伸到前沿技术动态和工业界最佳实践。"
        "思考过程请展示批判性思维和深度技术洞察。"
    ),
}


class PromptBuilder:
    """
    Prompt 构建器。

    LLM 只负责生成 thinking + output 两部分内容。
    system 字段由配置文件的 system_mapping 映射表提供。
    """

    @staticmethod
    def build_messages(item: DataItem, system_prompt: str) -> list[dict]:
        """
        构建标准 OpenAI messages 格式，要求 LLM 生成 thinking 和 output。

        Args:
            item: 原始数据条目
            system_prompt: 全局 system prompt（给 LLM 的角色设定）

        Returns:
            list[dict]: OpenAI messages 格式
        """
        diff_hint = DIFFICULTY_SUFFIX.get(item.difficulty, "")

        user_content = f"""【技术领域】{item.type}
【难度等级】{item.difficulty}
【问题】{item.instruction}{diff_hint}

请严格按照以下两段格式输出（每部分不少于 500 字，必须使用 Markdown 格式）：

## 思考过程
[使用 Markdown 格式的完整思考推理过程，要求：
- 使用 ### 小标题划分推理阶段（如：### 问题分析、### 知识检索、### 逐步推理、### 结论形成）
- 关键技术术语使用 **粗体** 突出
- 对比分析使用表格或列表展示
- 不少于 500 字]

## 最终答案
[使用 Markdown 格式的完整答案，要求：
- 使用 ### 小标题组织内容结构（如：### 核心理论、### 技术细节、### 代码示例、### 延伸思考）
- 代码示例使用 ``` 代码块包裹并标注语言
- 关键技术要点使用有序/无序列表展示
- 重点内容使用 **粗体** 强调
- 不少于 500 字]"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        return messages


class Generator:
    """
    核心生成引擎。

    功能:
    - 批量并发调用 LLM 生成 thinking + output
    - 解析 LLM 响应
    - 长度校验 + 自动重试
    """

    def __init__(self, config: Config):
        self.generation_config = config.generation
        self.quality_config = config.quality
        self.system_prompt = config.system_prompt
        self.client = LLMClient(config.llama_cpp, config.concurrency.max_workers)

    async def close(self):
        """清理资源"""
        await self.client.close()

    async def generate_batch(
        self,
        items: list[DataItem],
        progress_callback=None,
    ) -> list[GeneratedItem]:
        """
        批量生成（并发执行）。

        Args:
            items: 待生成的 DataItem 列表
            progress_callback: 每条生成完成后的回调 callback(item)

        Returns:
            list[GeneratedItem]: 生成结果列表
        """
        tasks = [self.generate_one(item) for item in items]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        generated: list[GeneratedItem] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"生成异常 [{items[i].source_file}#{items[i].id}]: {result}")
                generated.append(GeneratedItem(
                    data_item=items[i],
                    passed=False,
                    error_message=str(result),
                ))
            else:
                generated.append(result)

            if progress_callback:
                progress_callback(generated[-1])

        return generated

    async def generate_one(self, item: DataItem) -> GeneratedItem:
        """
        生成单条数据（含自动重试）。

        Args:
            item: 原始数据条目

        Returns:
            GeneratedItem: 生成结果
        """
        messages = PromptBuilder.build_messages(item, self.system_prompt)
        temperature = self.client.get_temperature(
            item.difficulty, self.generation_config.temperature
        )

        best_result = GeneratedItem(data_item=item, passed=False)

        for attempt in range(self.quality_config.max_retry_per_item):
            try:
                t0 = time.time()
                raw_text = await self.client.chat(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=self.generation_config.max_tokens,
                    top_p=self.generation_config.top_p,
                    seed=self.generation_config.seed,
                )
                elapsed = time.time() - t0

                thinking, output = self._parse_response(raw_text)

                result = GeneratedItem(
                    data_item=item,
                    thinking=thinking,
                    output=output,
                    thinking_chars=len(thinking),
                    output_chars=len(output),
                    retry_count=attempt,
                    generation_time=elapsed,
                )

                # 长度校验
                if self._validate(result):
                    result.passed = True
                    logger.debug(
                        f"[{item.source_file}#{item.id}] 生成成功 "
                        f"(thinking={result.thinking_chars}字, "
                        f"output={result.output_chars}字, "
                        f"耗时={elapsed:.1f}s)"
                    )
                    return result

                # 长度不足，记录并重试
                logger.warning(
                    f"[{item.source_file}#{item.id}] 长度不足 "
                    f"(thinking={result.thinking_chars}字, "
                    f"output={result.output_chars}字), "
                    f"第 {attempt + 1} 次重试..."
                )
                best_result = result

            except Exception as e:
                logger.warning(
                    f"[{item.source_file}#{item.id}] 生成失败 "
                    f"(第 {attempt + 1} 次): {e}"
                )
                best_result.error_message = str(e)

        # 所有重试均失败
        if best_result.thinking_chars > 0:
            best_result.passed = (
                best_result.thinking_chars >= self.quality_config.min_thinking_chars
                and best_result.output_chars >= self.quality_config.min_output_chars
            )
        return best_result

    def _parse_response(self, raw_text: str) -> tuple[str, str]:
        """
        从 LLM 响应中提取 thinking 和 output。

        解析策略（按优先级）：
        1. Markdown 标题: "## 思考过程" + "## 最终答案"
        2. 英文标题: "## Thinking" + "## Output/Answer"
        3. 中文冒号标签: "思考过程：" + "最终答案："
        4. 兜底：按段落中点分割

        Args:
            raw_text: LLM 返回的完整文本

        Returns:
            tuple[str, str]: (thinking, output)
        """
        if not raw_text:
            return ("", "")

        # 策略 1: 中文 Markdown 标题
        thinking, output = self._split_by_headers(raw_text, [
            (r"##\s*思考过程", r"##\s*最终答案"),
            (r"##\s*思考过程", r"##\s*最终输出"),
        ])
        if thinking and output:
            return self._clean_pair(thinking, output)

        # 策略 2: 英文 Markdown 标题
        thinking, output = self._split_by_headers(raw_text, [
            (r"##\s*[Tt]hinking", r"##\s*[Oo]utput"),
            (r"##\s*[Tt]hinking", r"##\s*[Aa]nswer"),
        ])
        if thinking and output:
            return self._clean_pair(thinking, output)

        # 策略 3: 中文冒号标签
        thinking, output = self._split_by_labels(raw_text, [
            (r"思考过程[：:]", r"最终答案[：:]"),
            (r"思考[：:]", r"答案[：:]"),
        ])
        if thinking and output:
            return self._clean_pair(thinking, output)

        # 策略 4: 兜底分割
        thinking, output = self._fallback_split(raw_text)
        return self._clean_pair(thinking, output)

    def _split_by_headers(
        self, text: str, header_pairs: list[tuple[str, str]]
    ) -> tuple[str, str]:
        """按 Markdown 标题对分割。"""
        for thinking_pattern, output_pattern in header_pairs:
            t_match = re.search(thinking_pattern, text)
            if not t_match:
                continue
            o_match = re.search(output_pattern, text[t_match.end():])
            if not o_match:
                continue

            o_start = t_match.end() + o_match.start()
            thinking_content = text[t_match.end():o_start]
            output_content = text[o_start + len(o_match.group()):]

            next_section = re.search(r"\n##\s+", output_content)
            if next_section:
                output_content = output_content[:next_section.start()]

            if thinking_content.strip() and output_content.strip():
                return (thinking_content.strip(), output_content.strip())

        return ("", "")

    def _split_by_labels(
        self, text: str, label_pairs: list[tuple[str, str]]
    ) -> tuple[str, str]:
        """按中文标签分割。"""
        for thinking_pattern, output_pattern in label_pairs:
            t_match = re.search(thinking_pattern, text)
            if not t_match:
                continue
            o_match = re.search(output_pattern, text[t_match.end():])
            if not o_match:
                continue

            o_start = t_match.end() + o_match.start()
            thinking_content = text[t_match.end():o_start]
            output_content = text[o_start + len(o_match.group()):]

            if thinking_content.strip() and output_content.strip():
                return (thinking_content.strip(), output_content.strip())

        return ("", "")

    def _fallback_split(self, text: str) -> tuple[str, str]:
        """兜底分割：按段落中点切分。"""
        # 尝试在 "---" 或 "###" 处分割
        for marker in ["\n---\n", "\n### ", "\n\n\n"]:
            parts = text.split(marker, 1)
            if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                return (parts[0].strip(), parts[1].strip())

        # 按双换行段落数中点分割
        paragraphs = re.split(r"\n\n+", text)
        if len(paragraphs) >= 2:
            mid = len(paragraphs) // 2
            thinking = "\n\n".join(paragraphs[:mid])
            output = "\n\n".join(paragraphs[mid:])
            return (thinking.strip(), output.strip())

        # 按句子均分
        sentences = re.split(r"(?<=[。！？.!?])\s*", text)
        if len(sentences) >= 4:
            mid = max(len(sentences) // 2, 1)
            thinking = "".join(sentences[:mid])
            output = "".join(sentences[mid:])
            return (thinking.strip(), output.strip())

        # 极端：字数均分
        half = len(text) // 2
        return (text[:half].strip(), text[half:].strip())

    @staticmethod
    def _clean_pair(thinking: str, output: str) -> tuple[str, str]:
        """清理 thinking/output 文本。"""
        thinking = re.sub(r"^[\s\n]+", "", thinking)
        thinking = re.sub(r"^[#*\-—=]+\s*", "", thinking)
        output = re.sub(r"^[\s\n]+", "", output)
        output = re.sub(r"^[#*\-—=]+\s*", "", output)
        return (thinking, output)

    def _validate(self, item: GeneratedItem) -> bool:
        """质量校验：检查 thinking 和 output 长度是否达标。"""
        return (
            item.thinking_chars >= self.quality_config.min_thinking_chars
            and item.output_chars >= self.quality_config.min_output_chars
        )
