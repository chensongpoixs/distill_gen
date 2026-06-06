"""
配置加载与管理模块。

配置优先级: YAML 文件 > Config dataclass 默认值
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ============================================================
# 子配置 dataclass
# ============================================================

@dataclass
class LlamaCppConfig:
    """llama.cpp 服务连接配置"""
    base_url: str = "http://localhost:8080/v1"
    api_key: str = "not-needed"
    model: str = "qwen3-4b"
    timeout: int = 300
    max_retries: int = 3
    retry_delay: float = 2.0


@dataclass
class TemperatureConfig:
    """难度分层温度配置"""
    primary: float = 0.3       # 初级
    intermediate: float = 0.5  # 中级
    advanced: float = 0.7      # 高级


@dataclass
class GenerationConfig:
    """LLM 生成参数配置"""
    max_tokens: int = 4096
    top_p: float = 0.9
    seed: int = 42
    temperature: TemperatureConfig = field(default_factory=TemperatureConfig)


@dataclass
class DataConfig:
    """数据路径配置"""
    input_dir: str = "."
    output_dir: str = "./gen_output"
    checkpoint_dir: str = "./gen_output/checkpoints"


@dataclass
class QualityConfig:
    """质量控制配置"""
    min_thinking_chars: int = 500
    min_output_chars: int = 500
    max_retry_per_item: int = 3


@dataclass
class ConcurrencyConfig:
    """并发控制配置"""
    max_workers: int = 3


@dataclass
class FieldsConfig:
    """字段映射配置"""
    required: list = field(default_factory=lambda: ["type", "instruction"])
    optional: list = field(default_factory=lambda: ["system", "difficulty", "input", "thinking", "output"])


@dataclass
class SystemMapping:
    """
    System 角色映射配置。

    根据配置的 fixed_value 或数据的 type 字段查找 system 角色描述。
    优先级: fixed_value（非空）→ overrides[type] → default_template.format(type=type)
    """
    default_template: str = "你是{type}领域的资深技术专家。"
    overrides: dict = field(default_factory=dict)
    fixed_value: str = ""  # 非空时所有条目统一使用该固定值，忽略 type 映射

    def get_system(self, type_value: str) -> str:
        """
        获取 system 角色描述。

        Args:
            type_value: 技术领域标签（如 "vLLM", "深度学习"）

        Returns:
            str: system 角色描述
        """
        # 如果配置了固定值，直接返回（不走映射表）
        if self.fixed_value:
            return self.fixed_value
        if type_value in self.overrides:
            return self.overrides[type_value]
        return self.default_template.format(type=type_value)


# ============================================================
# 主配置
# ============================================================

@dataclass
class Config:
    """蒸馏数据生成系统总配置"""
    llama_cpp: LlamaCppConfig = field(default_factory=LlamaCppConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)
    fields: FieldsConfig = field(default_factory=FieldsConfig)
    system_mapping: SystemMapping = field(default_factory=SystemMapping)
    system_prompt: str = (
        "你是 AI Infra 领域的资深技术专家，精通音视频编解码、流媒体传输、"
        "GPU/CUDA 编程、深度学习推理框架等技术栈。"
    )


# ============================================================
# 配置加载
# ============================================================

def _deep_update(base: object, override: dict) -> None:
    """递归地将 override dict 中的值合并到 base dataclass 中。"""
    for key, value in override.items():
        if key not in base.__dataclass_fields__:
            continue
        if isinstance(value, dict):
            nested = getattr(base, key)
            if hasattr(nested, "__dataclass_fields__"):
                _deep_update(nested, value)
                continue
        setattr(base, key, value)


def load_config(config_path: Optional[str] = None) -> Config:
    """
    加载并合并配置。

    优先级: YAML 文件 > Config 默认值

    Args:
        config_path: YAML 配置文件路径。为 None 时仅返回默认配置。

    Returns:
        Config: 合并后的配置对象。
    """
    config = Config()

    if config_path is None:
        return config

    config_file = Path(config_path)
    if not config_file.exists():
        import warnings
        warnings.warn(f"配置文件不存在: {config_path}，使用默认配置")
        return config

    try:
        import yaml
    except ImportError:
        import warnings
        warnings.warn("PyYAML 未安装，使用默认配置。安装: pip install pyyaml")
        return config

    with open(config_file, "r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)

    if yaml_data is None:
        return config

    # 递归合并各子配置
    section_map = {
        "llama_cpp": "llama_cpp",
        "generation": "generation",
        "data": "data",
        "quality": "quality",
        "concurrency": "concurrency",
        "fields": "fields",
    }

    for yaml_key, attr_name in section_map.items():
        if yaml_key in yaml_data and isinstance(yaml_data[yaml_key], dict):
            _deep_update(getattr(config, attr_name), yaml_data[yaml_key])

    # system_prompt 是顶层简单字段
    if "system_prompt" in yaml_data:
        config.system_prompt = yaml_data["system_prompt"]

    # system_mapping 配置
    if "system_mapping" in yaml_data:
        sm = yaml_data["system_mapping"]
        if isinstance(sm, dict):
            if "default_template" in sm:
                config.system_mapping.default_template = sm["default_template"]
            if "overrides" in sm and isinstance(sm["overrides"], dict):
                config.system_mapping.overrides = sm["overrides"]
            if "fixed_value" in sm:
                config.system_mapping.fixed_value = sm["fixed_value"]

    return config


def resolve_path(path_str: str, base_dir: Optional[Path] = None) -> Path:
    """
    将相对路径转为绝对路径。

    Args:
        path_str: 路径字符串
        base_dir: 基准目录，默认为配置文件所在目录

    Returns:
        Path: 绝对路径
    """
    p = Path(path_str)
    if p.is_absolute():
        return p
    if base_dir:
        return (base_dir / p).resolve()
    return p.resolve()
