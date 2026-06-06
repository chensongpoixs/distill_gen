# 蒸馏数据生成系统

## 1. 项目概述

### 1.1 背景

当前 `ai_infra_audio_video/` 目录下共有 **17 个 JSON 文件**，总计约 **16,000+ 行数据**，覆盖 AI Infra（音视频/流媒体/GPU/CUDA/vLLM/深度学习）领域。每条数据包含 `system`、`type`、`difficulty`、`instruction`、`thinking`、`output` 等字段。

**现状问题**：现有 `thinking` 和 `output` 字段内容过于简短（多数 < 100 字），缺乏深度推理链和详细解答，不利于高质量知识蒸馏训练。

### 1.2 目标

利用本地 llama.cpp 的 OpenAI-compatible HTTP API，基于每条数据的 `system`、`type`、`difficulty`、`instruction` 字段，重新生成 **高质量、长文本**（>500 字）的 `thinking`（思维链推理过程）和 `output`（最终答案），以 **Markdown 格式** 保存至可配置的输出目录。

### 1.3 核心约束

| 约束 | 说明 |
|------|------|
| 生成字段 | `thinking` ≥ 500 字，`output` ≥ 500 字 |
| API 协议 | OpenAI-compatible `/v1/chat/completions` |
| 输出格式 | Markdown 文件（`.md`），人类可读 |
| 配置化 | llama.cpp 服务信息 + 保存目录均可通过 YAML 配置 |
| 可恢复 | 支持断点续传（checkpoint），避免重复生成 |

---

## 2. 业界蒸馏数据生成最佳实践

### 2.1 知识蒸馏数据类型

参考 DeepSeek-R1、Qwen3、LLaMA-Factory 等业界方案，蒸馏数据分为以下几种：

| 类型 | 说明 | 适用场景 |
|------|------|---------|
| **CoT (Chain-of-Thought)** | 逐步推理链，展示中间思考过程 | 数学、逻辑、代码 |
| **Self-Refine** | 多轮自我修正，先生成再批判改进 | 写作、翻译、复杂 QA |
| **Instruction-Following** | 遵循指令的直接回答 | 通用对话 |
| **Multi-turn Dialog** | 多轮对话交互 | 聊天、客服 |
| **Tool-Use** | 工具调用与推理 | Agent 场景 |

本项目聚焦 **CoT 蒸馏**（领域知识问答），核心生成 "思考过程(thinking)" + "最终回答(output)"。

### 2.2 Prompt 工程策略（业界参考）

#### 2.2.1 System Prompt 设计原则

参考 DeepSeek-R1 和 Qwen3 的蒸馏方案：

```
你是 AI Infra 领域的资深技术专家，精通音视频编解码、流媒体传输、
GPU/CUDA 编程、深度学习推理优化、vLLM/TensorRT 部署等技术栈。

你的回答必须：
1. 先展示完整的思考推理过程（thinking），再给出最终答案（output）
2. thinking 部分需包含：问题分析 → 知识检索 → 逐步推理 → 结论形成
3. output 部分需包含：核心答案 + 技术细节 + 实践建议 + 延伸知识点
4. 每个部分至少 500 字，确保内容深度和完整性
```

#### 2.2.2 Few-shot 示例策略

业界研究表明，提供 2-3 个高质量示例可显著提升生成质量：

- **Positive Example**: 展示目标格式和深度标准
- **Negative Example**: 展示应避免的简短/敷衍回答
- **Edge Case Example**: 展示边界情况的处理

#### 2.2.3 难度分层生成策略

| 难度 | thinking 要求 | output 要求 | 温度 |
|------|-------------|------------|------|
| 初级 | 概念解释 + 基础推理 | 定义 + 原理 + 简单示例 | 0.3 |
| 中级 | 问题分解 + 对比分析 + 权衡 | 详细原理 + 代码示例 + 工程考量 | 0.5 |
| 高级 | 深度分析 + 架构权衡 + 批判性思维 | 全面方案 + 性能对比 + 前沿动态 | 0.7 |

### 2.3 质量控制机制

参考 LIMA、Orca、WizardLM 等蒸馏数据集的质控方案：

```
输入数据 → 格式校验 → LLM 生成 → 长度检查 → 质量评分 → 重试/丢弃 → 输出
                                ↓ (< 500字)
                              重试(最多3次)
                                ↓ (仍不合格)
                              降级标记 + 记录日志
```

- **长度检查**：thinking ≥ 500 字，output ≥ 500 字
- **格式检查**：Markdown 结构完整性
- **去重检查**：与已有数据语义相似度 < 0.85（BERTScore）
- **人工抽检**：每批次随机抽取 5% 人工审核

### 2.4 批量生成与并发控制

参考工业界方案（Alpaca、Vicuna）：

```
┌─────────────┐     ┌──────────────┐     ┌──────────────────┐
│  Data Queue  │────▶│  Worker Pool  │────▶│  Result Collector │
│  (asyncio)   │     │  (并发 N 个)   │     │  (checkpoint)     │
└─────────────┘     └──────┬───────┘     └──────────────────┘
                           │
                    ┌──────▼───────┐
                    │  llama.cpp   │
                    │  HTTP Server │
                    └──────────────┘
```

- **并发数**：2-4（本地 llama.cpp 服务，避免 GPU OOM）
- **超时**：单请求 300s
- **重试**：指数退避（1s → 2s → 4s → 8s），最多 3 次
- **限流**：Semaphore 控制并发 + asyncio 异步 I/O

---

## 3. 系统架构

### 3.1 整体架构

```
ai_infra_audio_video/
├── distill_gen/                    # 蒸馏数据生成系统
│   ├── __init__.py                 # 模块入口
│   ├── config.py                   # 配置加载与管理
│   ├── loader.py                   # JSON 数据加载器
│   ├── llm_client.py               # llama.cpp HTTP 客户端
│   ├── generator.py                # 核心生成引擎
│   ├── writer.py                   # Markdown 输出器
│   └── pipeline.py                 # 主流程编排
├── config_gen.yaml                 # 生成配置文件
├── gen_output/                     # 输出目录（可配置）
│   ├── checkpoints/                # 断点续传 checkpoint
│   ├── deeplearn_llm_0_100/        # 每个源文件一个子目录
│   │   ├── 1001_深度学习_中级.md    # 单条数据 Markdown
│   │   ├── 1002_神经网络_高级.md
│   │   └── ...
│   ├── gpt-vllm_0_10/
│   └── ...
└── DISTILL_DATA_GEN_DESIGN.md      # 本文档
```

### 3.2 模块职责

| 模块 | 文件 | 职责 |
|------|------|------|
| 配置层 | `config.py` + `config_gen.yaml` | 加载/合并配置，支持 YAML + 环境变量覆盖 |
| 数据层 | `loader.py` | 扫描 JSON 文件 → 解析 → 提取有效字段 → 去重 |
| 客户端 | `llm_client.py` | OpenAI-compatible HTTP 客户端，封装重试/超时/限流 |
| 生成层 | `generator.py` | Prompt 构建 → LLM 调用 → 响应解析 → 长度验证 |
| 输出层 | `writer.py` | Markdown 格式化 → 文件写入 → 索引生成 |
| 编排层 | `pipeline.py` | 串联全流程 → 断点续传 → 进度报告 → 统计汇总 |

### 3.3 数据流

```
config_gen.yaml ──▶ config.py (Config dataclass)
                           │
                    ┌──────▼──────┐
                    │   loader.py  │
                    │  扫描 .json  │
                    │  解析每条记录 │
                    └──────┬──────┘
                           │ List[DataItem]
                    ┌──────▼──────┐
                    │ generator.py │
                    │  构建 Prompt  │
                    │  调用 LLM     │
                    │  解析响应     │
                    │  校验长度     │
                    └──────┬──────┘
                           │ List[GeneratedItem]
                    ┌──────▼──────┐
                    │  writer.py   │
                    │  Markdown 写 │
                    │  索引生成    │
                    └──────┬──────┘
                           │
                     gen_output/
```

---

## 4. 配置设计 (`config_gen.yaml`)

```yaml
# ============================================================
# 蒸馏数据生成系统 — 配置文件
# ============================================================

# --- llama.cpp 服务配置 ---
llama_cpp:
  base_url: "http://localhost:8080/v1"       # llama.cpp server 地址
  api_key: "not-needed"                      # llama.cpp 通常不需要 API key
  model: "qwen3-4b"                          # 模型名称（需与 server 端一致）
  timeout: 300                                # 单请求超时（秒）
  max_retries: 3                              # 最大重试次数
  retry_delay: 2.0                            # 重试基础延迟（秒）

# --- 生成参数 ---
generation:
  max_tokens: 4096                            # 单次生成最大 token 数
  top_p: 0.9                                  # nucleus sampling
  seed: 42                                    # 随机种子（可复现）
  
  # 难度分层温度
  temperature:
    primary: 0.3       # 初级
    intermediate: 0.5  # 中级
    advanced: 0.7      # 高级

# --- 数据路径 ---
data:
  input_dir: "."                              # JSON 文件所在目录（当前目录）
  output_dir: "./gen_output"                  # Markdown 输出根目录
  checkpoint_dir: "./gen_output/checkpoints"  # 断点续传目录

# --- 质量约束 ---
quality:
  min_thinking_chars: 500                     # thinking 最少字符数
  min_output_chars: 500                       # output 最少字符数
  max_retry_per_item: 3                       # 单条数据最大重试次数
  enable_bertscore_dedup: false               # BERTScore 去重（需 GPU）
  sample_review_ratio: 0.05                   # 随机抽检比例

# --- 并发控制 ---
concurrency:
  max_workers: 3                              # 最大并发请求数
  requests_per_minute: 20                     # 每分钟最大请求数（限流）

# --- 字段映射 ---
fields:
  required: ["type", "difficulty", "instruction"]  # 必需字段
  optional: ["system", "input", "thinking", "output"]  # 可选字段
  system_default: "你是 AI Infra 领域的资深技术专家。"  # system 默认值

# --- System Prompt 模板 ---
system_prompt: |
  你是 AI Infra 领域的资深技术专家，精通：
  - 音视频编解码（H.264/H.265/AV1、AAC/Opus）
  - 流媒体传输（WebRTC/RTMP/HLS/QUIC）
  - GPU/CUDA 编程与优化
  - 深度学习推理框架（vLLM/TensorRT-LLM/llama.cpp）
  - Transformer/SSM/CNN 架构设计
  - 分布式训练与推理系统
  - 视频/音频 AI 模型部署

  你的回答包含两部分：
  
  ## 思考过程（thinking）
  - 先分析问题的核心要点和考察意图
  - 梳理相关背景知识和技术原理
  - 展示逐步推理过程（含权衡分析）
  - 不少于 500 字
  
  ## 最终答案（output）
  - 给出结构清晰、内容完整的答案
  - 包含核心理论 + 工程实践 + 代码示例（如适用）
  - 延伸相关知识点和面试追问方向
  - 不少于 500 字
```

---

## 5. 核心模块详细设计

### 5.1 数据模型

```python
# distill_gen/loader.py

@dataclass
class DataItem:
    """原始数据条目"""
    id: int
    type: str              # 技术领域（vLLM/深度学习/神经网络/...）
    difficulty: str         # 难度（初级/中级/高级）
    instruction: str        # 问题/指令
    system: str = ""        # 系统提示
    input: str = ""         # 额外输入
    source_file: str = ""   # 来源 JSON 文件名
    source_checksum: str = ""  # 原始内容 hash（去重键）

@dataclass
class GeneratedItem:
    """生成结果"""
    data_item: DataItem
    thinking: str           # 生成的思维链
    output: str             # 生成的答案
    thinking_chars: int     # thinking 字符数
    output_chars: int       # output 字符数
    retry_count: int = 0    # 重试次数
    generation_time: float = 0.0  # 生成耗时（秒）
    passed: bool = True     # 是否通过质量检查
    error_message: str = "" # 错误信息（如有）
```

### 5.2 LLM 客户端

```python
# distill_gen/llm_client.py

class LLMClient:
    """
    OpenAI-compatible HTTP 客户端，专为 llama.cpp 优化。
    基于 httpx + asyncio 实现异步并发请求。
    
    核心方法：
    - chat(messages, temperature, max_tokens) -> str
      发送聊天请求，返回生成的文本
    - chat_structured(messages, temperature, max_tokens) -> dict
      返回结构化 JSON（thinking + output 分离）
    """
    
    def __init__(self, config: LlamaCppConfig):
        self.base_url = config.base_url
        self.api_key = config.api_key
        self.model = config.model
        self.timeout = config.timeout
        self.max_retries = config.max_retries
        self._semaphore = asyncio.Semaphore(
            # 从 concurrency config 读取
        )
    
    async def chat(self, messages: list[dict], **kwargs) -> str:
        """带重试和超时的异步聊天请求"""
        for attempt in range(self.max_retries):
            try:
                async with self._semaphore:
                    response = await self._send_request(messages, **kwargs)
                return response
            except (httpx.TimeoutException, httpx.HTTPError) as e:
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # 指数退避
```

### 5.3 Prompt 构建器

```python
# distill_gen/generator.py

class PromptBuilder:
    """
    根据难度分层构建不同的 Prompt 模板。
    
    初级：要求概念解释 + 简单示例
    中级：要求对比分析 + 代码示例
    高级：要求架构权衡 + 前沿动态
    """
    
    # 难度分层 prompt 后缀
    DIFFICULTY_SUFFIX = {
        "初级": "请用通俗易懂的语言解释基础概念，配合简单示例说明。",
        "中级": "请进行对比分析，说明技术选型考量，并提供代码示例。",
        "高级": "请深入分析架构设计决策，讨论性能优化方案，并延伸前沿发展。",
    }
    
    def build_messages(self, item: DataItem, system_prompt: str) -> list[dict]:
        """
        构建标准 messages 格式，包含：
        1. system prompt（全局技术专家角色）
        2. user message（type + difficulty + instruction + 格式要求）
        """
        diff_hint = self.DIFFICULTY_SUFFIX.get(item.difficulty, "")
        user_content = f"""【技术领域】{item.type}
【难度等级】{item.difficulty}
【问题】{item.instruction}
{diff_hint}

请严格按照以下格式输出（每部分不少于 500 字）：

## 思考过程
[你的完整思考推理过程：问题分析 → 知识检索 → 逐步推理 → 结论形成]

## 最终答案
[你的最终答案：核心理论 + 技术细节 + 代码示例（如适用）+ 延伸知识点]"""
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
```

### 5.4 生成引擎

```python
# distill_gen/generator.py

class Generator:
    """
    核心生成引擎。
    
    工作流程：
    1. 从 loader 获取 DataItem 列表
    2. 过滤已有 checkpoint（跳过已生成的数据）
    3. 使用 asyncio.gather 并发调用 LLMClient
    4. 解析 LLM 响应，提取 thinking/output
    5. 长度校验（≥ min_chars）
    6. 不合格项重试（最多 max_retry_per_item 次）
    7. 实时写入 checkpoint 防丢失
    """
    
    async def generate_batch(
        self,
        items: list[DataItem],
        config: GenerationConfig,
    ) -> list[GeneratedItem]:
        """批量生成，支持断点续传"""
        pass
    
    def _parse_response(self, raw_text: str) -> tuple[str, str]:
        """
        从 LLM 响应中提取 thinking 和 output。
        
        解析策略（按优先级）：
        1. 匹配 "## 思考过程" / "## 最终答案" Markdown 标题分割
        2. 匹配 "thinking:" / "output:" 键值对
        3. JSON 解析 {"thinking": ..., "output": ...}
        4. 兜底：将前半部分作为 thinking，后半部分作为 output
        """
        pass
    
    def _validate(self, item: GeneratedItem) -> bool:
        """质量校验：长度 + 格式"""
        if item.thinking_chars < self.min_thinking_chars:
            return False
        if item.output_chars < self.min_output_chars:
            return False
        return True
```

### 5.5 Markdown 输出器

```python
# distill_gen/writer.py

class MarkdownWriter:
    """
    Markdown 格式输出器。
    
    文件命名规则：
        {id}_{type}_{difficulty}.md
    
    目录结构：
        gen_output/
        ├── {source_filename_without_ext}/
        │   ├── 0001_vLLM_初级.md
        │   ├── 0002_vLLM_中级.md
        │   └── ...
        ├── _index.md                    # 总索引
        └── _stats.json                  # 生成统计
    """
    
    def write_item(self, item: GeneratedItem, output_dir: Path) -> Path:
        """将单条生成结果写入 Markdown 文件"""
        pass
    
    def write_index(self, all_items: list[GeneratedItem], output_dir: Path):
        """生成总索引文件，按 source_file + type + difficulty 分组"""
        pass
    
    def write_stats(self, stats: dict, output_dir: Path):
        """写入生成统计 JSON"""
        pass
```

### 5.6 Markdown 文件模板

```markdown
---
id: 1003
type: 强化学习
difficulty: 高级
source_file: deeplearn_llm_0_100.json
generation_time: 23.5s
retry_count: 0
passed: true
---

# 在大模型RLHF中为什么使用PPO而不是TRPO？

## 思考过程

[生成的 thinking 内容，≥ 500 字]

---

## 最终答案

[生成的 output 内容，≥ 500 字]

---

*生成配置：temperature=0.7, max_tokens=4096, model=qwen3-4b*
*生成时间：2026-06-06 12:00:00*
```

---

## 6. 主流程编排

```python
# distill_gen/pipeline.py

async def main(config_path: str = "config_gen.yaml"):
    """
    主流程：
    
    1. 加载配置 (config.py)
    2. 初始化各模块
    3. 扫描并加载 JSON 数据 (loader.py)
    4. 加载 checkpoint，过滤已生成条目
    5. 分批并发调用 LLM 生成 (generator.py)
    6. 实时写入 Markdown 文件 (writer.py)
    7. 更新 checkpoint
    8. 质量校验 + 重试不合格项
    9. 生成索引和统计报告
    10. 打印汇总
    """
    
    # Step 1: 加载配置
    cfg = load_config(config_path)
    
    # Step 2: 初始化模块
    client = LLMClient(cfg.llama_cpp)
    loader = DataLoader(cfg.data)
    generator = Generator(client, cfg.generation, cfg.quality)
    writer = MarkdownWriter()
    
    # Step 3: 加载数据
    all_items = loader.load_all_json_files()
    logger.info(f"共加载 {len(all_items)} 条数据")
    
    # Step 4: Checkpoint 恢复
    checkpoint = Checkpoint.load(cfg.data.checkpoint_dir)
    pending_items = [item for item in all_items 
                     if item.source_checksum not in checkpoint.completed]
    logger.info(f"待处理: {len(pending_items)} / 已完成: {len(checkpoint.completed)}")
    
    # Step 5-7: 批量生成 + 写入
    results = []
    for batch in chunked(pending_items, cfg.concurrency.max_workers):
        generated = await generator.generate_batch(batch)
        for item in generated:
            writer.write_item(item, cfg.data.output_dir)
            checkpoint.mark_completed(item.data_item.source_checksum)
        results.extend(generated)
        checkpoint.save()
        
        # 进度报告
        logger.info(f"进度: {len(results)}/{len(pending_items)}")
    
    # Step 8-9: 输出索引和统计
    writer.write_index(results, cfg.data.output_dir)
    writer.write_stats(compute_stats(results), cfg.data.output_dir)
    
    # Step 10: 打印汇总
    print_summary(results)
```

---

## 7. 关键设计决策

| 决策 | 理由 |
|------|------|
| **asyncio + httpx 异步架构** | llama.cpp 是 I/O 密集型 HTTP 调用，asyncio 可最大化并发吞吐 |
| **难度分层温度** | 高级题用高温度增加多样性，初级题用低温度保证准确性 |
| **Markdown 格式输出** | 人类可读 + 易于后续处理（转为 Parquet/JSON）+ LLM 友好 |
| **按源文件分子目录** | 保持数据溯源清晰，便于增量更新和单独评测 |
| **Checkpoint 机制** | 16,000+ 条数据生成耗时可能数小时，断点续传至关重要 |
| **Semaphore 控制并发** | 本地 llama.cpp 显存有限，过多并发会导致 OOM |
| **指数退避重试** | 避免瞬间重试对服务端造成压力 |
| **Checksum 去重** | 基于原始数据内容 hash，避免重复生成相同或高度相似数据 |
| **YAML 外置配置** | 改 llama.cpp 地址/端口/模型名只需修改配置文件，无需改代码 |

---

## 8. 实现计划

### Phase 1: 基础设施（核心模块）

| 步骤 | 文件 | 内容 |
|------|------|------|
| 1.1 | `config_gen.yaml` | 创建配置文件，定义所有可配置项 |
| 1.2 | `distill_gen/config.py` | Config dataclass + `load_config()` YAML 加载 |
| 1.3 | `distill_gen/loader.py` | JSON 扫描 + 解析 + 字段提取 + checksum 去重 |
| 1.4 | `distill_gen/llm_client.py` | OpenAI-compatible 异步 HTTP 客户端 + 重试/超时/限流 |

### Phase 2: 生成核心

| 步骤 | 文件 | 内容 |
|------|------|------|
| 2.1 | `distill_gen/generator.py` | PromptBuilder + Generator + 响应解析 + 长度校验 |
| 2.2 | `distill_gen/writer.py` | MarkdownWriter + 文件命名 + 索引 + 统计 |

### Phase 3: 编排与质量

| 步骤 | 文件 | 内容 |
|------|------|------|
| 3.1 | `distill_gen/pipeline.py` | 主流程编排 + checkpoint + 进度报告 |
| 3.2 | `distill_gen/__init__.py` | 公共 API 导出 |

### Phase 4: 测试与验证

| 步骤 | 内容 |
|------|------|
| 4.1 | 单文件小批量测试（10 条数据验证全流程） |
| 4.2 | 质量抽检（人工审核 5% 生成结果） |
| 4.3 | 并发压力测试（验证 llama.cpp 稳定性） |
| 4.4 | 全量生成（16,000+ 条） |

---

## 9. 使用方式

```bash
# 1. 启动 llama.cpp server（需先启动）
llama-server -m /path/to/qwen3-4b.gguf --host 0.0.0.0 --port 8080

# 2. 编辑配置文件
vim config_gen.yaml   # 修改 llama_cpp.base_url / model / output_dir 等

# 3. 运行生成
python -m distill_gen.pipeline --config config_gen.yaml

# 4. 断点续传（直接重新运行即可，checkpoint 自动跳过已完成条目）
python -m distill_gen.pipeline --config config_gen.yaml

# 5. 查看结果
ls gen_output/
cat gen_output/_stats.json
```

---

## 10. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| llama.cpp 服务 OOM | 生成中断 | Semaphore 限流 + 分批处理 + 断点续传 |
| 生成内容质量低（< 500 字） | 数据不可用 | 自动重试 3 次 + 降级标记 + 日志记录 |
| 网络/服务异常 | 请求失败 | 指数退避重试 + 超时设置 + 错误日志 |
| 16,000+ 条生成耗时过长 | 工程进度 | asyncio 并发 + 断点续传 + 按优先级分批 |
| 生成内容格式不标准 | 解析失败 | 多重解析策略 + 兜底分割逻辑 |

---

## 11. 依赖清单

```txt
# requirements_gen.txt
pyyaml>=6.0          # YAML 配置解析
httpx>=0.27.0        # 异步 HTTP 客户端
tqdm>=4.66.0         # 进度条
rich>=13.0.0         # 终端美化输出
```

> 注：仅依赖 Python 标准库 + 上述轻量第三方库。不依赖 torch/transformers，确保在无 GPU 环境也可独立运行。
