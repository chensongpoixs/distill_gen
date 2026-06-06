"""
JSON 数据加载模块。

扫描指定目录下所有 .json 文件，解析数据条目，提取有效字段，去重。
"""

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from distill_gen.config import Config

logger = logging.getLogger(__name__)


@dataclass
class DataItem:
    """原始数据条目"""
    id: int
    type: str
    difficulty: str
    instruction: str
    system: str = ""
    input: str = ""
    source_file: str = ""
    source_checksum: str = ""

    def __hash__(self):
        return hash(self.source_checksum)


class DataLoader:
    """
    JSON 数据加载器。

    扫描 input_dir 下所有 .json 文件，解析每条数据，
    计算 source_checksum 用于去重和断点续传。
    """

    def __init__(self, config: Config):
        self.input_dir = Path(config.data.input_dir)
        self.fields_config = config.fields
        self._seen_checksums: set[str] = set()

    def load_all_json_files(self) -> list[DataItem]:
        """
        扫描并加载所有 JSON 文件。

        Returns:
            list[DataItem]: 所有有效数据条目（已去重）。
        """
        json_files = sorted(self.input_dir.glob("*.json"))
        if not json_files:
            logger.warning("未找到任何 .json 文件")
            return []

        logger.info(f"找到 {len(json_files)} 个 JSON 文件")

        all_items: list[DataItem] = []
        for json_file in json_files:
            items = self._load_single_file(json_file)
            all_items.extend(items)
            logger.info(f"  {json_file.name}: 加载 {len(items)} 条数据")

        logger.info(f"总计加载 {len(all_items)} 条有效数据（去重后）")
        return all_items

    def _load_single_file(self, file_path: Path) -> list[DataItem]:
        """加载单个 JSON 文件并解析所有条目。"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"JSON 解析失败: {file_path} — {e}")
            return []
        except Exception as e:
            logger.error(f"读取文件失败: {file_path} — {e}")
            return []

        if not isinstance(data, list):
            logger.warning(f"{file_path.name} 不是 JSON 数组，跳过")
            return []

        items = []
        for entry in data:
            item = self._parse_entry(entry, file_path.name)
            if item is not None:
                items.append(item)

        return items

    def _parse_entry(self, entry: dict, source_file: str) -> Optional[DataItem]:
        """
        解析单条数据字段。

        必需字段: type, instruction
        可选字段: difficulty (默认"中级"), system (默认""), input
        其他字段: id（自动生成默认值）

        Returns:
            DataItem 或 None（缺少必需字段时）
        """
        # 校验必需字段
        for field_name in self.fields_config.required:
            if field_name not in entry or not entry[field_name]:
                logger.debug(f"跳过条目（缺少字段 '{field_name}'）: {entry.get('id', '?')}")
                return None

        entry_id = entry.get("id", 0)
        if not isinstance(entry_id, int):
            try:
                entry_id = int(entry_id)
            except (ValueError, TypeError):
                entry_id = 0

        # difficulty 非必填，缺失时默认"中级"
        difficulty = entry.get("difficulty", "")
        if not difficulty:
            difficulty = "中级"

        system = entry.get("system", "")
        if not system:
            # DataItem.system 不再从原始数据取值，生成阶段由配置映射表填充
            system = ""

        item = DataItem(
            id=entry_id,
            type=str(entry["type"]).strip(),
            difficulty=difficulty,
            instruction=str(entry["instruction"]).strip(),
            system=str(system).strip(),
            input=str(entry.get("input", "")).strip(),
            source_file=source_file,
        )

        # 计算内容 checksum 用于去重
        item.source_checksum = self._compute_checksum(item)

        # 去重检查
        if item.source_checksum in self._seen_checksums:
            logger.debug(f"跳过重复条目: id={item.id}, checksum={item.source_checksum[:8]}")
            return None

        self._seen_checksums.add(item.source_checksum)
        return item

    def _compute_checksum(self, item: DataItem) -> str:
        """基于 type + difficulty + instruction 计算 MD5 哈希作为去重键。"""
        content = f"{item.type}|{item.difficulty}|{item.instruction}"
        return hashlib.md5(content.encode("utf-8")).hexdigest()
