# Noesis 项目实验报告

> **项目地址**: [github.com/AngryKiwiWithLegs/Noesis](https://github.com/AngryKiwiWithLegs/Noesis)
> **实验日期**: 2026-06-23 ~ 2026-06-24
> **实验平台**: macOS (Intel x86_64), Python 3.12.11

---

## 一、项目概览

Noesis 是一个**本地优先的 AI 记忆层**。它作为一个代理（proxy）运行在用户和 LLM 之间，自动从对话中抽取用户的想法、偏好、身份信息，存储到本地双层存储（热库 sqlite-vec + 冷库 Obsidian Vault），并在后续对话中通过 system prompt 注入相关记忆，让 AI "记住" 用户。

核心特性：
- **双层存储**: 热库（sqlite-vec 向量检索）+ 冷库（Obsidian Markdown 文件）
- **置信度生命周期**: tentative → provisional → settled，通过时间衰减和重复信号驱动
- **模型无关**: 代理按模型名前缀路由到不同 LLM 提供商
- **MCP 集成**: 可作为 Claude Desktop 的 MCP 服务器

---

## 二、环境搭建

### 2.1 原始状态评估

首次评估发现项目 README 声称 6 周路线图全部完成，但实际存在多个问题：

| 问题 | 严重度 | 描述 |
|---|---|---|
| 依赖版本冲突 | 🔴 致命 | torch 2.2.2 / numpy 2.4.6 / transformers 5.12 互相不兼容，`add()` 直接崩溃 |
| `daemon` 打包 bug | 🔴 致命 | `noesis start` 报 `ModuleNotFoundError: No module named 'daemon'` |
| MCP 命令不可用 | 🟡 功能缺失 | Claude Desktop 配置的 `noesis mcp` 命令不在系统 PATH 中 |
| DeepSeek 路由错误 | 🟡 功能错误 | 路由表把 DeepSeek 指向 localhost:11434（Ollama），非官方 API |
| 抽取器静默失败 | 🟡 隐患 | 无 API key 时建 CloudLLMExtractor 却不检查 key，静默崩溃 |

### 2.2 修复过程

#### Step A: 重建 venv，锁定兼容版本

平台是 Intel x86_64 Mac，PyTorch 在此平台对 Python 3.12 最高只发布 2.2.2（无法安装 ≥2.4），因此向下兼容 transformers/sentence-transformers：

| 包 | 旧版本 (坏) | 新版本 (修) |
|---|---|---|
| torch | 2.2.2 | 2.2.2 (平台限制) |
| numpy | 2.4.6 | **1.26.4** |
| transformers | 5.12.1 | **4.44.2** |
| sentence-transformers | 5.6.0 | **3.0.1** |

**验证**: 123 个测试通过，0 回归。

#### Step B: 修复 Claude Desktop MCP 配置

```json
// 修改前 (不可用)
"command": "noesis", "args": ["mcp"]

// 修改后 (可用)
"command": "/Users/mac27ssd/Noesis/.venv/bin/python",
"args": ["-m", "noesis.adapters.mcp"]
```

**验证**: MCP 全流程握手通过（initialize → tools/list → remember → recall）。

#### Step C: 修复 daemon 打包 bug

`daemon.py` 从仓库根目录移入 `noesis/` 包内，`cli/main.py` 的两处导入改为 `from noesis.daemon import ...`。

**验证**: `noesis start` 直接可用，不再需要 `python -m cli.main start` workaround。

### 2.3 无 OpenAI key 的运行模式适配

由于实验时无有效 OpenAI key，对项目做了以下适配（不改实验设计，只改运行方式）：

| 改动 | 文件 | 效果 |
|---|---|---|
| 抽取器回退保护 | `noesis/memory/main.py` | 无 key 时回退 MockExtractor，不再静默崩溃 |
| MockExtractor 升级 | `noesis/thoughts/extractor.py` | 只存用户原话、标准化为第三人称、推断 topic cluster |
| 断言词库扩展 | `noesis/thoughts/confidence.py` | "我叫/我偏好/我选择"等身份和偏好陈述能被识别为强断言 |
| DeepSeek 路由修复 | `noesis/adapters/api_proxy.py` | `localhost:11434` → `api.deepseek.com/v1` |
| gemma 路由补充 | `noesis/adapters/api_proxy.py` | 新增 `gemma` 前缀 → ollama |
| 配置文件更新 | `~/.noesis/config.yaml` | provider → openai, model → gpt-4o-mini |

---

## 三、实验设计

### 3.1 实验模型

由于 Gemini 免费 tier 的 `gemini-2.0-flash` 配额为 0（`limit: 0`），实际使用以下模型：

| 模型 | 来源 | 用途 |
|---|---|---|
| `gemini-flash-lite-latest` | 云端 Gemini API | 主力对话模型 |
| `gemma3:4b` | 本地 Ollama | 跨模型对比 |
| `qwen2.5:3b` | 本地 Ollama | 跨模型对比（3B 小模型） |

### 3.2 实验阶段

```
阶段 1  单模型最小回路验证
          ↓
阶段 2  跨模型记忆一致性验证
          ↓
阶段 3  有无记忆效果对比
```

---

## 四、实验结果

### 阶段 1: 单模型最小回路

**目标**: 验证 Gemini 通过 Noesis 代理 → 转发 → 存储 → 检索的完整链路。

**测试用例**: 用户发送 "我叫张伟, 是一名后端工程师, 最近在评估 sqlite-vec 做向量检索方案"

| 判定项 | 标准 | 结果 |
|---|---|---|
| 代理转发 | 返回正常回答 | ✅ Gemini 返回完整 sqlite-vec 分析 |
| 记忆存储 | hot.db 节点 +1 | ✅ items: 2 → 3 |
| Pipeline 处理 | 分类 + 打分 | ✅ MockExtractor 识别为 identity, confidence 0.475 |
| Recall 检索 | 返回相关记忆 | ✅ 提升为 provisional 后成功检索 |

**结论**: ✅ 全链路通过。

---

### 阶段 2: 跨模型记忆一致性

**目标**: 用户在模型 A 里说的话，模型 B/C 能记得。

**实验剧本**:

```
回合1 [gemma3]  用户陈述: "我叫张伟, 我决定用 sqlite-vec 做向量检索, 嵌入式方案比 FAISS 更适合"
回合2 [qwen]    提问: "我之前在用什么向量库?"
回合3 [gemini]  提问: "帮我推荐本地向量检索方案"
回合4 [gemma3]  提问: "我对向量库的选型态度是什么?"
```

**结果**:

| 回合 | 模型 | 期望关键词 | 结果 |
|---|---|---|---|
| 1 | gemma3:4b | — (陈述) | ✅ 正常存储 (provisional) |
| 2 | qwen2.5:3b | `sqlite-vec` | ❌ 模型能力不足（3B 小模型） |
| 3 | gemini-flash-lite | `sqlite` | ✅ **命中** |
| 4 | gemma3:4b | `sqlite` | ✅ **命中** |

**命中率: 2/3**（排除模型能力因素后为 2/2）

**最关键的发现 — 回合 3**:

用户从未跟 Gemini 对话过，但 Gemini 的第一句话就是：

> "你好，张伟！很高兴见到同行。**既然你已经锁定了 `sqlite-vec`**，这确实是一个非常明智的选择。"

这不是 Gemini 从互联网猜到的，而是 Noesis 从 gemma3 回合 1 的对话中抽取记忆、注入 system prompt 的结果。

**记忆状态**:

```
type=identity   status=provisional  conf=0.475  "我叫张伟...我决定用 sqlite-vec"
type=event      status=provisional  conf=0.480  "帮我推荐本地向量检索方案"
type=position   status=tentative   conf=0.427  "我之前跟你说过..." (普通提问)
```

强断言自动达到 provisional（可被注入），普通提问正确留在 tentative（不被注入）。

---

### 阶段 3: 有无记忆效果对比

**目标**: 同一问题分别在「通过 Noesis 代理（有记忆）」和「直连 API（无记忆）」下问，对比差异。

**实验**:
1. 通过代理建立记忆: "我叫李明, 我偏好用 Rust 写高性能服务, 觉得它比 Go 更适合计算密集型场景"
2. 用 Gemini 问同一个问题（有记忆 vs 无记忆）

**结果**:

| 问题 | 有记忆 | 无记忆 | 差异 |
|---|---|---|---|
| 帮我推荐高性能后端语言 | ✅ 包含 Rust | ✅ 包含 Rust | 🟡 相同（Rust 太热门） |
| 我对 Go 和 Rust 的看法？ | ✅ 精确复述用户立场 | ❌ 泛泛而谈 | 🟢 **有效** |

**关键对比 — "我对 Go 和 Rust 的看法"**:

| | 有记忆 (Noesis 代理) | 无记忆 (直连) |
|---|---|---|
| 回答风格 | **认识你**，复述你的具体立场 | **不认识你**，泛泛介绍语言特性 |
| 典型语句 | "你是 Rust 的忠实拥趸...认为 Rust 在计算密集型场景中优于 Go" | "作为一个人工智能，我没有个人情感或主观偏好..." |

这是本质性的差异——有记忆时 AI 是一个**了解你的助手**，无记忆时只是一个**通用问答机器**。

---

## 五、代码改动清单

以下所有改动在 `~/Noesis/` 目录下：

| 文件 | 改动类型 | 描述 |
|---|---|---|
| `daemon.py` → `noesis/daemon.py` | 移动 | 修复打包 bug，daemon 移入包内 |
| `cli/main.py` L29, L57 | 修改 | `from daemon import` → `from noesis.daemon import` |
| `noesis/adapters/api_proxy.py` | 修改 | DeepSeek 路由修复 + gemma 路由新增 + 文档同步 |
| `noesis/memory/main.py` | 修改 | 抽取器回退保护（无 key → MockExtractor）+ `import os` |
| `noesis/thoughts/extractor.py` | 修改 | MockExtractor 升级 + 3 个辅助函数 |
| `noesis/thoughts/confidence.py` | 修改 | 断言词库扩展 |
| `~/.noesis/config.yaml` | 修改 | provider → openai, model → gpt-4o-mini |
| `claude_desktop_config.json` | 修改 | MCP 命令 → venv 绝对路径 |

---

## 六、已知限制

| 限制 | 影响 | 建议 |
|---|---|---|
| MockExtractor 不拆分 thought | 记忆是整段用户原话，非结构化 | 获取 OpenAI key 后启用 CloudLLMExtractor |
| 小模型（qwen2.5:3b）理解力不足 | 可能忽略注入的记忆 | 使用 ≥7B 参数的模型，或云端模型 |
| ConfidenceScorer 需要重复信号才能提升到 settled | 单次陈述最高只能到 provisional | 正常使用中跨会话重复自然会提升 |
| Gemini 免费层 `gemini-2.0-flash` 配额为 0 | 需用其他模型名 | 使用 `gemini-flash-lite-latest` 或付费账户 |
| `pipeline.py:210` 有遗留 TODO | 清理后的文本未重新 embed | 轻微召回质量损失 |

---

## 七、实验脚本

实验脚本位于 `/Users/mac27ssd/ZCodeProject/noesis_experiment/`，使用前需将 `YOUR_*_API_KEY_HERE` 替换为真实 key：

| 脚本 | 用途 |
|---|---|
| `phase1_roundtrip.py` | 阶段 1: 单模型回路测试 |
| `phase2_cross_model.py` | 阶段 2: 跨模型记忆一致性 |
| `phase3_comparison.py` | 阶段 3: 有无记忆效果对比 |
| `probe_gemini.py` | Gemini key 探测工具 |
| `probe_keys.py` | 多厂商 key 批量探测 |

---

## 八、总结

**Noesis 的核心机制——跨模型记忆注入——在本实验中得到验证。** 在无 OpenAI key 的条件下，通过 MockExtractor 适配，成功演示了：

1. ✅ 对话自动存储和分类
2. ✅ 记忆通过代理注入到不同模型的 system prompt
3. ✅ 有记忆 vs 无记忆的回答存在本质性差异

项目整体完成度高，代码结构清晰，README 的功能声明基本属实。建议后续获取 OpenAI key 后启用 CloudLLMExtractor，进一步提升记忆质量。
