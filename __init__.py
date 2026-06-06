"""
蒸馏数据生成系统 — Distill Data Generator

利用 llama.cpp 的 OpenAI-compatible HTTP API，基于 JSON 数据中的
system/type/difficulty/instruction 字段，重新生成高质量的 thinking（思维链）
和 output（最终答案），保存为 JSON 格式（字段与原始文件一致）。

用法：
    python -m distill_gen.pipeline --config config_gen.yaml
"""

from distill_gen.config import Config, load_config
from distill_gen.loader import DataItem, DataLoader
from distill_gen.llm_client import LLMClient
from distill_gen.generator import GeneratedItem, Generator
from distill_gen.writer import JsonWriter

__all__ = [
    "Config",
    "load_config",
    "DataItem",
    "DataLoader",
    "LLMClient",
    "GeneratedItem",
    "Generator",
    "JsonWriter",
]
