"""
对话模式生成引擎模块。

负责:
- Prompt 构建（要求 LLM 返回 JSON 数组，每轮含 instruction/thinking/output/follow_up）
- 调用 LLMClient 进行文本生成
- 解析 LLM 响应（JSON 数组解析，支持截断兜底）
- 长度校验与重试
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field

from distill_gen.config import Config
from distill_gen.llm_client import LLMClient
from distill_gen.loader import DataItem

logger = logging.getLogger(__name__)


@dataclass
class DialogueRound:
    """单轮对话内容"""
    round: int = 0
    instruction: str = ""
    thinking: str = ""
    output: str = ""
    follow_up_instruction: str = ""


@dataclass
class DialogueGeneratedItem:
    """对话模式生成结果"""
    data_item: DataItem
    system: str = ""
    rounds: list[DialogueRound] = field(default_factory=list)
    system_chars: int = 0
    num_rounds: int = 0
    total_thinking_chars: int = 0
    total_output_chars: int = 0
    retry_count: int = 0
    generation_time: float = 0.0
    passed: bool = True
    error_message: str = ""


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


class DialoguePromptBuilder:
    """对话模式 Prompt 构建器。"""

    @staticmethod
    def build_messages(
        item: DataItem,
        system_prompt: str,
        num_rounds: int,
    ) -> list[dict]:
        diff_hint = DIFFICULTY_SUFFIX.get(item.difficulty, "")

        user_content = (
            f"【技术领域】{item.type}\n"
            f"【难度等级】{item.difficulty}\n"
            f"【初始问题】{item.instruction}\n"
            f"【目标轮次】{num_rounds} 轮\n"
            f"\n"
            f"请生成一个{item.difficulty}级别的多轮技术对话，共 {num_rounds} 轮。\n"
            f"\n"
            f"要求：\n"
            f"1. 第 1 轮的 instruction 为上面给定的初始问题\n"
            f"2. 从第 2 轮起，每轮的 instruction 为上一轮的 follow_up_instruction\n"
            f"3. 每轮的结构：\n"
            f"   - instruction: 本轮问题\n"
            # f"   - thinking: 思维推理过程（Markdown 格式，使用 ### 小标题、**粗体**、列表、表格等，不少于 500 字）\n"
            f"     - thinking 字段必须模拟顶级技术专家的**分步推理过程**，而不仅仅是写一段长文。请严格按以下结构撰写，每部分使用 ### 小标题划分，并使用 **粗体**、- 列表等增强可读性：\n"
            f"              \n"
            f"				**Step 1: 理解与复述 (Understand & Restate)**\n"
            f"				- 用 2-3 句话提炼本轮 instruction 的核心诉求。\n"
            f"				- 如果是第 2 轮及以后，必须**明确提及上一轮 output 中的关键结论或遗留问题**，体现上下文承接。\n"
            f"				- 确认问题中是否包含隐含约束、特定环境或性能要求。\n"
            f"				\n"
            f"				**Step 2: 知识检索与分析 (Knowledge Retrieval & Analysis)**\n"
            f"				- 列出解决此问题需要的核心技术概念、原理或工具（如特定的库、算法、设计模式）。\n"
            f"				- 解释这些概念之间的关联，必要时引用官方文档定义或经典案例。\n"
            f"				- 若涉及对比，用表格列出不同方案的核心差异：\n"
            f"				  | 方案 | 优点 | 缺点 | 适用场景 |\n"
            f"				  |------|------|------|----------|\n"
            f"				  | A    | ...  | ...  | ...      |\n"
            f"				\n"
            f"				**Step 3: 方案构思与选择 (Solution Design & Selection)**\n"
            f"				- 提出至少两种可能的回答方向或解决方案。\n"
            f"				- 分析每种方案的优劣，并明确说明**最终选择哪种**及其理由（如：更符合用户场景、更高效、更易维护）。\n"
            f"				- 对于复杂问题，给出思维导图式的拆解：\n"
            f"				  - 主方案\n"
            f"				    - 子步骤 1\n"
            f"				    - 子步骤 2\n"
            f"				\n"
            f"				**Step 4: 验证与组织 (Verification & Structuring)**\n"
            f"				- 对选定的方案进行自我检查：逻辑是否自洽？代码是否能运行？边界情况是否覆盖？\n"
            f"				- 思考如何将答案组织成用户友好的 Markdown 格式：\n"
            f"				  - 开头是否需要一个 TL;DR 总结？\n"
            f"				  - 代码块是否需要注释？\n"
            f"				  - 是否需要添加表格进行参数说明？\n"
            f"				- 确认输出将严格达到 500 字以上的要求，且无冗余空话。\n"
            f"				\n"
            f"   - output: 最终答案 : \n"
			f"		- 是面向用户的最终回答，必须用 Markdown 撰写，结构清晰。\n"
			f"		- 必须包含以下五个小节（作为三级标题）：\n"
			f"		  ### 1. 核心要点\n"
			f"		  ### 2. 详细解析\n"
			f"		  ### 3. 代码示例\n"
			f"		  ### 4. 注意事项\n"
			f"		  ### 5. 延伸思考\n"
			f"		- 「代码示例」中必须使用 ```语言 包裹的代码块，代码应有详细注释，变量命名有意义。\n"
			f"		- 「详细解析」中须包含原理阐述、方案对比（如有）、边界与异常说明。\n"
			f"		- 「延伸思考」中可提示后续优化方向、相关技术栈或进阶话题。\n"
			f"		- 全文不少于 500 字，禁止出现“作为 AI”等非技术套话，禁止直接复制 `thinking` 中的内容。\n"
			f"		- 根据难度等级，output 还需满足以下附加要求：\n"
			f"		  - 初级：请用通俗易懂的语言解释基础概念，配合简单示例帮助理解。\n"
			f"		  - 中级：请进行对比分析，说明技术选型的工程考量，并提供代码/伪代码示例。\n"
			f"		  - 高级：请深入分析架构设计决策，讨论性能优化方案，延伸到前沿技术动态和工业界最佳实践。\n"
            f"   - follow_up_instruction: 基于本轮回答提出的下一个深入问题\n"
            f"4. follow_up_instruction 要自然承接，逐步深入，形成递进式对话链\n"
            f"5. 每轮的 thinking 和 output 均不少于 500 字"
            f"{diff_hint}\n"
            f"\n"
            f"请返回如下格式的 JSON 数组（不要输出其他内容）：\n"
            f"\n"
            f"```json\n"
            f"[\n"
            f"  {{\n"
            f'    "round": 1,\n'
            f'    "instruction": "...",\n'
            f'    "thinking": "...",\n'
            f'    "output": "...",\n'
            f'    "follow_up_instruction": "..."\n'
            f"  }},\n"
            f"  ...\n"
            f"]\n"
            f"```"
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]


class DialogueGenerator:
    """对话模式生成引擎。"""

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
        num_rounds: int = 15,
    ) -> list[DialogueGeneratedItem]:
        tasks = [self.generate_one(item, num_rounds) for item in items]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        generated: list[DialogueGeneratedItem] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    f"对话生成异常 [{items[i].source_file}#{items[i].id}]: {result}"
                )
                generated.append(DialogueGeneratedItem(
                    data_item=items[i],
                    passed=False,
                    error_message=str(result),
                ))
            else:
                generated.append(result)

        return generated

    async def generate_one(
        self,
        item: DataItem,
        num_rounds: int = 15,
    ) -> DialogueGeneratedItem:
        messages = DialoguePromptBuilder.build_messages(
            item, self.system_prompt, num_rounds
        )
        temperature = self.client.get_temperature(
            item.difficulty, self.generation_config.temperature
        )

        # 请求详情日志
        logger.debug("=" * 60)
        logger.debug(f"请求 [{item.source_file}#{item.id}] "
                     f"{item.type}/{item.difficulty} | "
                     f"temperature={temperature} | "
                     f"max_tokens={self.generation_config.max_tokens}")
        for msg in messages:
            logger.debug(f"  [{msg['role']}]: {msg['content']}")
        logger.debug("=" * 60)

        best_result = DialogueGeneratedItem(data_item=item, passed=False)

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

                # 响应详情日志
                logger.debug(f"响应 [{item.source_file}#{item.id}] "
                             f"耗时 {elapsed:.1f}s | "
                             f"长度 {len(raw_text)} 字符")
                logger.debug(f"原始回复:\n{raw_text}")
                if len(raw_text) > 2000:
                    logger.debug(f"... (截断, 共 {len(raw_text)} 字)")
                logger.debug("-" * 60)

                rounds = self._parse_response(raw_text)

                result = DialogueGeneratedItem(
                    data_item=item,
                    rounds=rounds,
                    num_rounds=len(rounds),
                    total_thinking_chars=sum(len(r.thinking) for r in rounds),
                    total_output_chars=sum(len(r.output) for r in rounds),
                    retry_count=attempt,
                    generation_time=elapsed,
                )

                if self._validate(result):
                    result.passed = True
                    logger.debug(
                        f"[{item.source_file}#{item.id}] 对话生成成功 "
                        f"({len(rounds)} 轮, "
                        f"thinking={result.total_thinking_chars}字, "
                        f"output={result.total_output_chars}字, "
                        f"耗时={elapsed:.1f}s)"
                    )
                    return result

                logger.warning(
                    f"[{item.source_file}#{item.id}] 对话轮次不足或长度不够 "
                    f"({len(rounds)} 轮, "
                    f"thinking={result.total_thinking_chars}字, "
                    f"output={result.total_output_chars}字), "
                    f"第 {attempt + 1} 次重试..."
                )
                result.passed = True
                    # logger.debug(
                    #     f"[{item.source_file}#{item.id}] 对话生成成功 "
                    #     f"({len(rounds)} 轮, "
                    #     f"thinking={result.total_thinking_chars}字, "
                    #     f"output={result.total_output_chars}字, "
                    #     f"耗时={elapsed:.1f}s)"
                    # )
                return result
                #best_result = result

            except Exception as e:
                logger.warning(
                    f"[{item.source_file}#{item.id}] 对话生成失败 "
                    f"(第 {attempt + 1} 次): {e}"
                )
                best_result.error_message = str(e)

        if best_result.rounds:
            best_result.passed = (
                len(best_result.rounds) >= 1
                and all(
                    len(r.thinking) >= self.quality_config.min_thinking_chars
                    and len(r.output) >= self.quality_config.min_output_chars
                    for r in best_result.rounds
                )
            )
        return best_result

    def _parse_response(self, raw_text: str) -> list[DialogueRound]:
        """从 LLM content 文本中提取对话轮次数组。三级策略。"""
        if not raw_text:
            return []

        # —— 策略 1: 括号栈匹配（直接提取 JSON 数组）——
        array = self._extract_json_array(raw_text)
        if array:
            rounds = self._to_dialogue_rounds(array)
            if len(rounds) >= 1:
                return rounds

        # —— 策略 2: 贪婪正则 ```json ... ``` ——
        match = re.search(r'```(?:json)?\s*\n(.*)```', raw_text, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
            array = self._safe_json_loads(json_str)
            if not array:
                array = self._safe_json_loads(self._repair_json(json_str))
            if isinstance(array, list):
                rounds = self._to_dialogue_rounds(array)
                if len(rounds) >= 1:
                    return rounds

        # —— 策略 3: 部分 JSON 提取（响应被截断时）——
        partial = self._extract_partial_json_array(raw_text)
        if len(partial) >= 1:
            return self._to_dialogue_rounds(partial)

        logger.warning(f"对话 JSON 解析失败，无法提取有效轮次")
        return []

    def _to_dialogue_rounds(self, array: list) -> list[DialogueRound]:
        """将解析出的 JSON 数组转为 DialogueRound 列表，过滤无效轮次。"""
        rounds = []
        for entry in array:
            if not isinstance(entry, dict):
                continue
            if not all(k in entry for k in ("instruction", "thinking", "output")):
                continue
            rounds.append(DialogueRound(
                round=entry.get("round", len(rounds) + 1),
                instruction=str(entry.get("instruction", "")).strip(),
                thinking=str(entry.get("thinking", "")).strip(),
                output=str(entry.get("output", "")).strip(),
                follow_up_instruction=str(entry.get("follow_up_instruction", "")).strip(),
            ))
        return rounds

    def _extract_json_array(self, text: str) -> list | None:
        """从文本中提取完整 JSON 数组（括号栈匹配）。"""
        start = text.find("[")
        if start < 0:
            return None

        depth = 0
        in_string = False
        escape_next = False

        for i in range(start, len(text)):
            ch = text[i]
            if escape_next:
                escape_next = False
                continue
            if ch == "\\" and not escape_next:
                escape_next = True
                continue
            if ch == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        return None
        return None

    def _extract_partial_json_array(self, text: str) -> list:
        """从被截断的残缺 JSON 数组中提取尽可能多的完整轮次。"""
        start = text.find("[")
        if start < 0:
            return []

        results = []
        i = start + 1

        while i < len(text):
            while i < len(text) and text[i] in " \n\r\t,":
                i += 1
            if i >= len(text) or text[i] == "]":
                break

            if text[i] == "{":
                obj_end = self._match_bracket(text, i, "{", "}")
                if obj_end > i:
                    try:
                        obj = json.loads(text[i : obj_end + 1])
                        if isinstance(obj, dict) and all(
                            k in obj for k in ("instruction", "thinking", "output")
                        ):
                            results.append(obj)
                    except json.JSONDecodeError:
                        repaired = self._repair_json(text[i : obj_end + 1])
                        try:
                            obj = json.loads(repaired)
                            if isinstance(obj, dict) and all(
                                k in obj for k in ("instruction", "thinking", "output")
                            ):
                                results.append(obj)
                        except json.JSONDecodeError:
                            pass
                    i = obj_end + 1
                else:
                    break
            else:
                i += 1

        return results

    def _match_bracket(self, text: str, start: int, open_ch: str, close_ch: str) -> int:
        """匹配括号，返回闭合位置（未闭合时返回 -1）。"""
        depth = 0
        in_string = False
        escape_next = False

        for i in range(start, len(text)):
            ch = text[i]
            if escape_next:
                escape_next = False
                continue
            if ch == "\\":
                escape_next = True
                continue
            if ch == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    return i
        return -1

    @staticmethod
    def _safe_json_loads(json_str: str):
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
            if ch == "\\":
                escape_next = True
                result.append(ch)
                continue
            if ch == '"':
                if in_string:
                    rest = json_str[i + 1 :].lstrip()
                    if rest and rest[0] in ",}:":
                        in_string = False
                    else:
                        result.append("\\")
                else:
                    in_string = True
                result.append(ch)
                continue
            if in_string and ch == "\n":
                result.append("\\n")
                continue
            if in_string and ch == "\r":
                result.append("\\r")
                continue
            if in_string and ch == "\t":
                result.append("\\t")
                continue
            result.append(ch)

        fixed = "".join(result)
        fixed = re.sub(r",\s*}", "}", fixed)
        fixed = re.sub(r",\s*]", "]", fixed)
        return fixed

    def _validate(self, item: DialogueGeneratedItem) -> bool:
        """校验对话生成结果。至少 1 轮且每轮的 thinking/output 达标。"""
        if not item.rounds:
            return False
        for r in item.rounds:
            if len(r.thinking) < self.quality_config.min_thinking_chars:
                return False
            if len(r.output) < self.quality_config.min_output_chars:
                return False
        return True
