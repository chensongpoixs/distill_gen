# DistillGen — 大规模知识蒸馏数据合成引擎

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)

**DistillGen** 是一个面向 SFT（Supervised Fine-Tuning）的高质量蒸馏数据批量合成框架。它从原始 JSON 题库出发，调用大模型 API 自动生成包含深度推理链（Chain-of-Thought）和结构化答案的蒸馏数据，广泛兼容 **GPT-4o / Claude / Gemini / llama.cpp / vLLM / Ollama** 等主流模型服务。

> 灵感来源：DeepSeek-R1 蒸馏管线、Orca/ WizardLM 渐进式蒸馏、LIMA 少样本高质量对齐、OpenHermes 多领域指令数据工程。

---

## 1. 核心特性

| 特性 | 说明 |
|------|------|
| 🌐 **多 Provider 兼容** | 任何 OpenAI-compatible `/v1/chat/completions` 端点均可接入（OpenAI / Anthropic / Gemini / llama.cpp / vLLM / Ollama / LiteLLM） |
| 🧠 **CoT 推理链蒸馏** | 每条数据生成 `reasoning_content`（≥500 字思维链）+ `content`（≥500 字最终答案），内容为 Markdown 结构化格式 |
| 🎯 **难度分层生成** | 初级 / 中级 / 高级 三档，分别配置不同的 Temperature 和 Prompt 策略 |
| 📝 **JSON 原生输出** | 要求 LLM 返回严格 JSON 结构，通过栈匹配 + `json.loads` 无损解析，杜绝内容截断 |
| 🔁 **断点续传** | 基于 MD5 checksum 的 Checkpoint 机制，中断后重启自动跳过已完成条目 |
| ⚡ **异步并发** | `asyncio` + `httpx` + `Semaphore` 控制并发，指数退避自动重试 |
| 📊 **统计报告** | 生成 `_stats.md`，含耗时分布（min/median/P95/max）、字数统计、难度/领域明细 |
| 🔧 **全 YAML 配置** | Provider / 模型 / 路径 / 温度 / System 映射表全部通过单一配置文件管理 |

---

## 2. 架构总览

```
                  ┌──────────────────────────┐
                  │     config_gen.yaml       │  ← 统一配置入口
                  │  provider / model / temp   │
                  │  path / system_mapping    │
                  └────────────┬─────────────┘
                               │
  ┌──────────────┐     ┌──────▼──────┐     ┌──────────────┐     ┌──────────────┐
  │  loader.py   │────▶│ generator.py │────▶│  writer.py   │────▶│  gen_output/ │
  │  JSON 扫描    │     │  Prompt构建   │     │  JSON 分组写入 │     │  *.json      │
  │  去重+校验    │     │  LLM 并发调用  │     │  _stats.md   │     │  _stats.md   │
  └──────────────┘     │  响应解析+校验 │     └──────────────┘     └──────────────┘
                       └──────┬──────┘
                              │
                       ┌──────▼──────┐
                       │ llm_client   │
                       │ httpx +      │
                       │ asyncio +     │
                       │ 指数退避重试   │
                       └──────┬──────┘
                              │
               ┌──────────────┼──────────────┐
               ▼              ▼              ▼
          ┌─────────┐  ┌──────────┐  ┌───────────┐
          │ GPT-4o  │  │  Claude  │  │  Gemini   │
          │ (OpenAI)│  │(Anthropic)│  │ (Google)  │
          └─────────┘  └──────────┘  └───────────┘
               │              │              │
               └──────────────┼──────────────┘
                              │
               ┌──────────────▼──────────────┐
               │ OpenAI-Compatible Endpoint  │
               │ /v1/chat/completions        │
               │ (llama.cpp / vLLM / Ollama) │
               └─────────────────────────────┘
```

---

## 3. 数据流与字段映射

```
输入 JSON (原始)                    输出 JSON (蒸馏后)
┌─────────────────────┐            ┌─────────────────────────────────┐
│ {                   │            │ {                               │
│   "id": 1,          │   ──▶     │   "id": 1,        ← 原样保留     │
│   "type": "vLLM",   │   ──▶     │   "type": "vLLM",  ← 原样保留    │
│   "difficulty":"高级"│  ──▶     │   "difficulty":"高级"← 原样保留   │
│   "instruction":"?" │   ──▶     │   "instruction":"?"← 原样保留    │
│   "input": "",      │   ──▶     │   "input": "",     ← 原样保留    │
│   "thinking": "...",│   ✕       │   "system": "You are...",← 配置表 │
│   "output": "..."   │   ✕       │   "thinking": "...",← LLM 生成  │
│ }                   │            │   "output": "..."   ← LLM 生成  │
└─────────────────────┘            └─────────────────────────────────┘
```

- `id` / `type` / `difficulty` / `instruction` / `input` → **原样保留**
- `system` → 从 `system_mapping` 配置表按 `type` 匹配（或设为全局固定值如 `"You are a helpful assistant."`）
- `thinking` / `output` → **LLM 蒸馏生成**（JSON 格式返回，内容为 Markdown）

---

## 4. 多 Provider 配置

### 4.1 GPT-4o / GPT-4.1 (OpenAI)

```yaml
llama_cpp:
  base_url: "https://api.openai.com/v1"
  api_key: "sk-xxxxxxxxxxxxxxxxxxxxxxxx"
  model: "gpt-4o"
  timeout: 120
  max_retries: 3
  retry_delay: 2.0
```

### 4.2 Claude 3.5 / Claude Opus 4 (Anthropic)

> 可通过 LiteLLM 代理或 Anthropic 官方 OpenAI-compatible 端点接入。

```yaml
llama_cpp:
  base_url: "https://api.anthropic.com/v1"     # Anthropic Messages API (OpenAI-compatible)
  api_key: "sk-ant-xxxxxxxxxxxxxxxxxxxxxxxx"
  model: "claude-sonnet-4-6"
  timeout: 300
  max_retries: 3
  retry_delay: 5.0
```

> 或通过本地 LiteLLM 代理统一接入多个 provider：
> ```bash
> litellm --model anthropic/claude-sonnet-4-6 --port 4000
> ```
> ```yaml
>   base_url: "http://localhost:4000/v1"
> ```

### 4.3 Gemini 2.5 (Google)

```yaml
llama_cpp:
  base_url: "https://generativelanguage.googleapis.com/v1beta/openai"
  api_key: "AIzaSyxxxxxxxxxxxxxxxxxxxxxxxxx"
  model: "gemini-2.5-pro"
  timeout: 180
  max_retries: 3
  retry_delay: 3.0
```

### 4.4 本地部署 (llama.cpp / vLLM / Ollama)

```yaml
# llama.cpp (GGUF)
llama_cpp:
  base_url: "http://localhost:8080/v1"
  api_key: "not-needed"
  model: "qwen3-4b"

# vLLM
llama_cpp:
  base_url: "http://localhost:8000/v1"
  api_key: "not-needed"
  model: "Qwen/Qwen3-4B"

# Ollama
llama_cpp:
  base_url: "http://localhost:11434/v1"
  api_key: "ollama"
  model: "qwen3:4b"
```

> **核心原理**：以上所有 Provider 均通过标准 OpenAI Chat Completions 协议通信，只需修改 `base_url` / `api_key` / `model` 即可无缝切换。

---

## 5. System 角色字段策略

`system` 字段不通过 LLM 生成，而是由配置文件控制，支持三种模式：

| 模式 | 配置 | 效果 |
|------|------|------|
| **固定值** | `fixed_value: "You are a helpful assistant."` | 所有条目统一使用该值 |
| **Type 映射** | `fixed_value: ""` + 配置 `overrides[type]` | 按技术领域匹配角色描述 |
| **模板兜底** | `fixed_value: ""` + 未命中 overrides 时 | 使用 `default_template` 自动填充 |

```yaml
system_mapping:
  fixed_value: ""                               # 非空 = 全局固定值
  default_template: "你是{type}领域的资深技术专家。"
  overrides:
    vLLM: "你是vLLM推理框架的资深系统架构师，精通PagedAttention..."
    深度学习: "你是深度学习领域的算法研究员，精通神经网络架构..."
    # ... 28+ 个领域预定义
```

---

## 6. 快速开始

### 6.1 安装依赖

```bash
pip install pyyaml httpx
```

### 6.2 配置

```bash
cp config_gen.yaml.example config_gen.yaml
vim config_gen.yaml
```

最少需要修改：

```yaml
llama_cpp:
  base_url: "https://api.openai.com/v1"   # 你的 provider 端点
  api_key: "sk-xxxx"                       # API Key
  model: "gpt-4o"                          # 模型名

data:
  input_dir: "./data"                      # 原始 JSON 文件目录
  output_dir: "./gen_output"               # 输出目录
```

### 6.3 运行

```bash
# 全量生成
python -m distill_gen.pipeline --config config_gen.yaml

# 测试模式（仅 10 条）
python -m distill_gen.pipeline --config config_gen.yaml --limit 10

# 详细日志
python -m distill_gen.pipeline --config config_gen.yaml --verbose

# 断点续传（直接重新运行，自动跳过已完成条目）
python -m distill_gen.pipeline --config config_gen.yaml
```

### 6.4 查看结果

```bash
ls gen_output/

# 输出示例：
# ├── deeplearn_llm_0_100.json       # 蒸馏后的 JSON（与原始文件同名）
# ├── gpt-vllm_0_10.json
# ├── ...
# ├── _stats.md                       # 统计报告（Markdown）
# └── checkpoints/
#     └── completed.json              # 断点续传状态
```

---

## 7. LLM 输出格式

LLM 被要求返回严格 JSON：

```json
{
  "reasoning_content": "### 问题分析\n\n从问题本身出发...\n\n### 逐步推理\n\n1. **核心原理**: ...\n\n### 结论形成\n\n综合以上分析...",
  "content": "### 核心理论\n\n...\n\n### 技术细节\n\n...\n\n### 代码示例\n\n```python\n...\n```\n\n### 延伸思考\n\n..."
}
```

- `reasoning_content` → 映射为输出 JSON 的 `thinking` 字段
- `content` → 映射为输出 JSON 的 `output` 字段
- 内容均为 **Markdown 格式**（`###` 小标题、`**粗体**`、``` ` ``` 代码块、表格、列表）

解析采用 4 级兜底策略：

| 优先级 | 策略 | 方法 |
|--------|------|------|
| 1 | ` ```json {...} ``` ` 代码块 | 正则提取 + `json.loads()` |
| 2 | 直接 JSON `{"reasoning_content":...}` | 正则匹配 + `json.loads()` |
| 3 | 嵌入在文本中的完整 `{...}` | 括号深度栈匹配 + `json.loads()` |
| 4 | Markdown 标题分割 | 传统 `## 思考过程` / `## 最终答案` 分割 |

---

## 8. 质量控制管线

```
输入数据
  │
  ├─ 格式校验：type + instruction 必填，difficulty 缺失默认"中级"
  ├─ 去重：MD5(type|difficulty|instruction) 全局唯一
  │
  ▼
LLM 生成
  │
  ├─ JSON 解析：4 级兜底策略保证内容完整
  ├─ 长度校验：thinking ≥ 500 字 AND output ≥ 500 字
  ├─ 不合格 → 自动重试（指数退避，最多 3 次）
  │
  ▼
输出
  │
  ├─ JSON 文件：按 source_file 分组，字段与原始一致
  └─ _stats.md：耗时分布 + 字数统计 + 难度/领域明细
```

---

## 9. 关键设计决策

| 决策 | 理由 |
|------|------|
| **OpenAI-compatible 协议** | GPT-4o / Claude / Gemini / 本地模型 一个协议全部覆盖 |
| **JSON 原生输出** | `json.loads` 无损解析，避免正则截断导致的内容丢失 |
| **难度分层 Temperature** | 初级 0.3（稳定）→ 中级 0.5 → 高级 0.7（多样性） |
| **MD5 Checksum 去重** | 基于 `type|difficulty|instruction` 内容哈希，跨文件全局唯一 |
| **asyncio + Semaphore** | 最大化 HTTP I/O 吞吐，同时保护 API 端的并发限制 |
| **指数退避重试** | 1s → 2s → 4s → 8s，避免瞬时重试风暴 |
| **断点续传** | 每批次立即持久化 completed.json，支持随时中断恢复 |
| **YAML 全量配置** | 切换 Provider / 模型 / 路径只需修改配置文件，零代码改动 |

---

## 10. 模块清单

```
distill_gen/
├── __init__.py         # 公共 API 导出
├── config.py           # Config dataclass + YAML 加载 + SystemMapping
├── loader.py           # JSON 扫描 / 字段提取 / MD5 去重
├── llm_client.py       # OpenAI-compatible HTTP 客户端 (httpx + asyncio)
├── generator.py        # Prompt 构建 + LLM 调用 + JSON 解析 + 长度校验
├── writer.py           # JSON 分组输出 + _stats.md 统计报告
└── pipeline.py         # 主流程编排 + Checkpoint + 进度日志
```

---

## 11. 依赖

```txt
pyyaml>=6.0          # YAML 配置解析
httpx>=0.27.0        # 异步 HTTP 客户端
```

> 零额外依赖。不依赖 torch / transformers / openai SDK，在任何 Python 3.10+ 环境中可直接运行。

---

## 12. 参考与致谢

- **DeepSeek-R1** — 推理链蒸馏与冷启动数据工程
- **Orca / WizardLM** — 渐进式指令蒸馏与复杂度分层
- **LIMA** — "Less is More" 高质量少样本对齐范式
- **OpenHermes / UltraChat** — 多领域指令数据构造方法论
- **Alpaca / Vicuna** — 批量蒸馏数据生成管线设计
- **llama.cpp** — 本地高效推理基础设施
