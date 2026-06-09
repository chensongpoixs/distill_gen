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
        从 LLM content 文本中提取 thinking/output。四级策略：

          1. 括号栈匹配 —— 完整 JSON 对象提取，支持 repair
          2. 贪婪正则 ```json ... ``` —— 代码块提取，支持 repair
          3. 部分 JSON 提取 —— 响应被 max_tokens 截断时，从残缺 JSON 中尽力提取
          4. Markdown 标题分割 —— 最终兜底（非 JSON 格式的响应）
        """
        if not raw_text:
            return ("", "")

        # —— 策略 1: 括号栈匹配（最可靠）——
        key_pos = raw_text.find('"thinking"')
        if key_pos >= 0:
            start = raw_text.rfind('{', 0, key_pos)
            if start >= 0:
                json_str = self._extract_json(raw_text, start)
                if json_str:
                    data = self._safe_json_loads(json_str)
                    if not data:
                        data = self._safe_json_loads(self._repair_json(json_str))
                    if data:
                        thinking = data.get("thinking", "")
                        output = data.get("output", "")
                        if thinking and output:
                            return (thinking, output)

        # —— 策略 2: 贪婪正则 ```json ... ``` ——
        match = re.search(r'```(?:json)?\s*\n(.*)```', raw_text, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
            data = self._safe_json_loads(json_str)
            if not data:
                data = self._safe_json_loads(self._repair_json(json_str))
            if data:
                thinking = data.get("thinking", "")
                output = data.get("output", "")
                if thinking and output:
                    return (thinking, output)

        # —— 策略 3: 部分 JSON 提取（响应被截断时）——
        # 策略 1/2 失败可能是 LLM 输出被 max_tokens 截断导致 JSON 不完整
        # 此时尝试从残缺 JSON 中提取已有内容，好过回到 Markdown 分割
        if key_pos >= 0:  # 存在 "thinking" 键 → 铁定是 JSON 格式响应
            thinking, output = self._extract_partial_json(raw_text)
            if thinking and output:
                return (thinking, output)

        # —— 策略 4: Markdown 标题分割（最终兜底）——
        return self._fallback_markdown_split(raw_text)

    @staticmethod
    def _safe_json_loads(json_str: str) -> dict | None:
        """安全 JSON 解析，失败返回 None。"""
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None

    def _repair_json(self, json_str: str) -> str:
        """
        修复 LLM 输出 JSON 的常见格式错误：
          1. 字符串值内未转义的 ASCII 双引号（LLM 用作中文引号 "" 的替代）
          2. 字符串值内的原始换行/制表符（应转义为 \\n \\t）
          3. } 或 ] 前的尾随逗号
        """
        result = []
        in_string = False
        escape_next = False

        for i, ch in enumerate(json_str):
            if escape_next:
                escape_next = False
                result.append(ch)
                continue
            if ch == '\\':
                escape_next = True
                result.append(ch)
                continue
            if ch == '"':
                if in_string:
                    # 在字符串内遇到引号：检查是否为 JSON 结构终止符
                    # —— 键的终止符后跟 : ，值的终止符后跟 , 或 }
                    rest = json_str[i + 1:].lstrip()
                    if rest and rest[0] in ',}:':
                        in_string = False
                    else:
                        # 未转义的引号（如中文引号）→ 加反斜杠转义
                        result.append('\\')
                else:
                    in_string = True
                result.append(ch)
                continue
            # 字符串内的原始控制字符 → 转义
            if in_string and ch == '\n':
                result.append('\\n')
                continue
            if in_string and ch == '\r':
                result.append('\\r')
                continue
            if in_string and ch == '\t':
                result.append('\\t')
                continue
            result.append(ch)

        fixed = ''.join(result)
        # 去除尾随逗号
        fixed = re.sub(r',\s*}', '}', fixed)
        fixed = re.sub(r',\s*]', ']', fixed)
        return fixed

    def _extract_partial_json(self, text: str) -> tuple[str, str]:
        """
        从被截断的残缺 JSON 中尽力提取 thinking/output 值。

        LLM 响应可能被 max_tokens 截断，导致 JSON 不完整（缺少 } 或 ```）。
        此方法用简单状态机直接提取字符串字段值，不要求完整 JSON 结构。
        """
        thinking = self._extract_json_field(text, "thinking")
        output = self._extract_json_field(text, "output")
        return (thinking or "", output or "")

    def _extract_json_field(self, text: str, field: str) -> str | None:
        """
        从可能残缺的 JSON 文本中提取指定字段的字符串值。

        先找 "field": "，然后用状态机读取字符串值（处理 \\ 转义），
        遇到未转义的 " 时判断：后跟 , 或 } 即为值结束，否则当作内容内的引号。
        如果到文本末尾都未闭合，返回已读内容（截断场景）。
        """
        pattern = rf'"{field}"\s*:\s*"'
        match = re.search(pattern, text)
        if not match:
            return None

        start = match.end()
        result = []
        i = start
        while i < len(text):
            ch = text[i]
            if ch == '\\':
                if i + 1 < len(text):
                    nxt = text[i + 1]
                    if nxt == 'n':
                        result.append('\n')
                    elif nxt == 't':
                        result.append('\t')
                    elif nxt == 'r':
                        result.append('\r')
                    elif nxt in ('"', '\\', '/'):
                        result.append(nxt)
                    else:
                        result.append('\\' + nxt)
                    i += 2
                    continue
            elif ch == '"':
                # 字符串可能在此结束，检查后续字符
                rest = text[i + 1:].lstrip()
                if not rest or rest[0] in ',}:':
                    # 值结束（正常闭合、JSON 末尾截断、下一键开始）
                    return ''.join(result)
                # 未转义的引号（如中文引号）→ 当作内容
                result.append(ch)
                i += 1
                continue
            elif ch in '\n\r\t':
                # JSON 字符串内的原始控制字符 → 当作内容
                result.append(ch)
                i += 1
                continue
            result.append(ch)
            i += 1

        # 走到文本末尾而未闭合 → 截断场景，返回已读内容
        return ''.join(result) if result else None

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
