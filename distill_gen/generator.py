"""
核心生成引擎模块。

负责:
- Prompt 构建（要求 LLM 返回 JSON 格式）
- 调用 LLMClient 进行文本生成
- 解析 LLM 响应（从 JSON 中提取 reasoning_content / content）
- 长度校验与重试
"""

import asyncio
import json
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
    """Prompt 构建器。要求 LLM 回复 content 中放入 ```json 代码块。"""

    @staticmethod
    def build_messages(item: DataItem, system_prompt: str) -> list[dict]:
        diff_hint = DIFFICULTY_SUFFIX.get(item.difficulty, "")

        user_content = f"""【技术领域】{item.type}
【难度等级】{item.difficulty}
【问题】{item.instruction}{diff_hint}

请将你的思考推理和最终答案以 JSON 格式放入 ```json 代码块中返回：

```json
{{
  "thinking": "完整的思考推理过程（Markdown格式，使用###小标题、**粗体**、列表、表格等， 不少于500字）",
  "output": "完整的最终答案（Markdown格式，使用###小标题、```代码块```、**粗体**、列表等，不少于500字）"
}}
```"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        return messages


class Generator:
    """
    核心生成引擎。
    """

    def __init__(self, config: Config):
        self.generation_config = config.generation
        self.quality_config = config.quality
        self.system_prompt = config.system_prompt
        self.client = LLMClient(config.llama_cpp, config.concurrency.max_workers)

    async def close(self):
        await self.client.close()

    async def generate_batch(
        self,
        items: list[DataItem],
        progress_callback=None,
    ) -> list[GeneratedItem]:
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

                if self._validate(result):
                    result.passed = True
                    logger.debug(
                        f"[{item.source_file}#{item.id}] 生成成功 "
                        f"(thinking={result.thinking_chars}字, "
                        f"output={result.output_chars}字, "
                        f"耗时={elapsed:.1f}s)"
                    )
                    return result

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

        if best_result.thinking_chars > 0:
            best_result.passed = (
                best_result.thinking_chars >= self.quality_config.min_thinking_chars
                and best_result.output_chars >= self.quality_config.min_output_chars
            )
        return best_result

    def _parse_response(self, raw_text: str) -> tuple[str, str]:
        """
        从 LLM content 文本中提取 thinking/output。三级策略：

          1. 括号栈匹配 —— 用 "thinking" 定位 JSON 对象，回溯 {，栈匹配完整 JSON。
             不受 output 内 ```python 等嵌套 fence 影响，最鲁棒。
          2. 贪婪正则 ```json ... ``` —— (.*) 贪婪匹配到全文最后一个 ```，确保外层闭合。
          3. Markdown 标题分割 —— 最终兜底。
        """
        if not raw_text:
            return ("", "")

        # —— 策略 1: 括号栈匹配（最可靠）——
        # 大模型输出的 JSON 通常有换行/缩进，{"thinking" 不是连续串，
        # 所以用 "thinking" 定位，再回溯找到 {
        key_pos = raw_text.find('"thinking"')
        if key_pos >= 0:
            start = raw_text.rfind('{', 0, key_pos)
            if start >= 0:
                json_str = self._extract_json(raw_text, start)
                if json_str:
                    data = self._safe_json_loads(json_str)
                    if data:
                        thinking = data.get("thinking", "")
                        output = data.get("output", "")
                        if thinking and output:
                            return (thinking, output)

        # —— 策略 2: 贪婪正则 ```json ... ``` ——
        # (.*) 贪婪匹配 → 全文最后一个 ``` → 外层闭合 fence
        # 避免 output 内的 ```python 等嵌套代码块被误匹配
        match = re.search(r'```(?:json)?\s*\n(.*)```', raw_text, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
            data = self._safe_json_loads(json_str)
            if data:
                thinking = data.get("thinking", "")
                output = data.get("output", "")
                if thinking and output:
                    return (thinking, output)

        # —— 策略 3: Markdown 标题分割（最终兜底）——
        return self._fallback_markdown_split(raw_text)

    @staticmethod
    def _safe_json_loads(json_str: str) -> dict | None:
        """安全 JSON 解析，失败返回 None。"""
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None

    def _extract_json(self, text: str, start: int) -> str | None:
        """从 start 位置提取完整 JSON 字符串（括号栈匹配）。"""
        depth = 0
        in_string = False
        escape_next = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape_next:
                escape_next = False
                continue
            if ch == '\\':
                escape_next = True
                continue
            if ch == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch in '{[':
                depth += 1
            elif ch in '}]':
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        return None

    def _fallback_markdown_split(self, text: str) -> tuple[str, str]:
        """兜底：按 Markdown 标题分割。"""
        # 尝试 ## 思考过程 / ## 最终答案
        for tp, op in [
            (r"##\s*思考过程", r"##\s*最终答案"),
            (r"##\s*[Tt]hinking", r"##\s*[Oo]utput"),
        ]:
            t_match = re.search(tp, text)
            if not t_match:
                continue
            o_match = re.search(op, text[t_match.end():])
            if not o_match:
                continue

            o_start = t_match.end() + o_match.start()
            thinking = text[t_match.end():o_start].strip()
            output = text[o_start + len(o_match.group()):].strip()

            # 截断后续 ## 标题之后的内容
            next_sec = re.search(r"\n##\s+", output)
            if next_sec:
                output = output[:next_sec.start()]

            if thinking and output:
                return (thinking, output)

        # 按 --- 或段落中点分割
        for marker in ["\n---\n", "\n\n\n"]:
            parts = text.split(marker, 1)
            if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                return (parts[0].strip(), parts[1].strip())

        paragraphs = re.split(r"\n\n+", text)
        if len(paragraphs) >= 2:
            mid = len(paragraphs) // 2
            return (
                "\n\n".join(paragraphs[:mid]).strip(),
                "\n\n".join(paragraphs[mid:]).strip(),
            )

        half = len(text) // 2
        return (text[:half].strip(), text[half:].strip())

    def _validate(self, item: GeneratedItem) -> bool:
        return (
            item.thinking_chars >= self.quality_config.min_thinking_chars
            and item.output_chars >= self.quality_config.min_output_chars
        )
