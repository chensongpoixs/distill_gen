# DistillGen — AGENTS.md

## 启动命令

### 标准模式（单轮 thinking + output）

```bash
# 全量生成
python -m distill_gen.pipeline --config config_gen.yaml

# 测试模式（10 条）
python -m distill_gen.pipeline --config config_gen.yaml --limit 10

# 详细日志
python -m distill_gen.pipeline --config config_gen.yaml --verbose
```

### 对话模式（N 轮 thinking → output → follow-up）

```bash
# 15 轮对话生成
python -m distill_gen.dialogue_pipeline --config config_gen.yaml

# 指定轮次数（如 10 轮）
python -m distill_gen.dialogue_pipeline --config config_gen.yaml --num-rounds 10

# 测试模式
python -m distill_gen.dialogue_pipeline --config config_gen.yaml --limit 5 --verbose
```

对话模式下，LLM 单次调用返回 JSON 数组（15 轮），输出路径为 `{output_dir}/dialogue/`，统计报告为 `_dialogue_stats.md`。注意 `max_tokens` 需要足够大（建议 ≥ 16000）以容纳 15 轮内容。

依赖：`pip install pyyaml httpx`（零额外依赖，无需 torch/transformers/openai SDK）

## 架构

```
distill_gen/
├── __init__.py         # 公共 API 导出
├── config.py           # YAML → Config dataclass（deep-merge）
├── loader.py           # JSON 扫描 + MD5(type|difficulty|instruction) 去重
├── llm_client.py       # OpenAI-compatible HTTP 客户端（httpx + asyncio + Semaphore）
├── generator.py        # 标准模式：Prompt → LLM → 4级 JSON 解析 → 校验
├── dialogue_generator.py # 对话模式：Prompt → LLM → JSON 数组解析 → 校验
├── writer.py           # 分组 JSON 输出 + _stats.md + 对话模式写入
├── pipeline.py         # 标准模式主流程 + Checkpoint
└── dialogue_pipeline.py # 对话模式主流程 + Checkpoint
```

## 关键细节（容易遗漏）

### 配置怪癖
- 配置顶层 key 是 `llama_cpp:`，但实际支持所有 OpenAI-compatible 端点（OpenAI / Anthropic / Gemini / vLLM / Ollama / llama.cpp），只需改 `base_url`/`api_key`/`model`
- `system_prompt` 是顶层字段（不在 section 下），`load_config()` 特殊处理
- `system_mapping` 三层优先级：`fixed_value`（非空则统一使用）→ `overrides[type]` → `default_template.format(type=type)`
- 路径字段（`input_dir`/`output_dir`/`checkpoint_dir`）若为绝对路径则原样使用，相对路径用配置文件所在目录解析
- `quality.min_thinking_chars`/`min_output_chars` 默认 500

### 数据流
- 输入 JSON 必须是**数组**（`[]`），每个元素是对象
- 必需字段：`type`、`instruction`；`difficulty` 缺失时默认 `"中级"`
- 标准模式输出字段：`id`/`type`/`difficulty`/`system`/`instruction`/`input`/`thinking`/`output`
- 对话模式输出字段：`id`/`type`/`difficulty`/`messages`（OpenAI SFT 格式，system → user → assistant 交替）
- 对话模式中 `messages` 内 assistant 消息含 `reasoning_content`（thinking）和 `content`（output）两个字段
- `id` 不全为 int 时会被尝试转换，转换失败置 0

### LLM 调用
- `llm_client.chat()` 只取 `choices[0].message.content`，模型返回的 `reasoning_content` 字段自动丢弃
- 标准模式解析 4 级策略：括号栈匹配 → ` ```json ` 代码块 → 部分 JSON（截断兜底）→ Markdown 标题分割
- 对话模式解析 3 级策略：括号栈匹配 JSON 数组 → ` ```json ` 代码块 → 部分数组截断兜底
- 长度不足时自动重试（指数退避 `retry_delay * 2^attempt`，最多 3 次），重试策略在 `llm_client.py` 和 `generator.py` 各有一套（LLM 请求级 + 条目级）
- 难度温度：初级 0.3、中级 0.5、高级 0.7

### 断点续传
- 基于 MD5 checksum，自动跳过 `checkpoints/completed.json` 中已标记条目
- 直接重新运行即可续传，无需额外参数
- 若所有条目已完成，会调用 `_rebuild_json_from_checkpoint()`，但此时不会写入 JSON 输出文件，仅生成统计报告，且统计中的字数不可用（不含 thinking/output）

### 已知限制
- `docs/README.md` 不相关（深度学习笔记），勿引用
- 无测试框架、无 CI 配置、无 lint/typecheck 配置
- `.gitignore` 只排除了 `__pycache__`
