# Phase1 底座候选：硬短板 × 创新切口 × 主底座决议

| 项 | 内容 |
|----|------|
| 文档 | `docs/phase1-base-fork-gaps-innovation.md` |
| 状态 | **已拍板口径**（分析意见成文）；不改业务代码；不修改冻结协议 |
| 日期 | 2026-11-03（UTC+8） |
| 效力 | 冲突以 `PHASE1-ARENA-PROTOCOL-v0.1.md` 为准；新颖点口径对齐 `docs/phase1-novelty-risks.md` / `docs/phase1-bf16-compute-strategy.md` |
| 范围 | 仅本仓（GitHub `Ultimate-XDU/An-operating-system-for-AI-models-and-tensor-inference`） |
| 触发 | 月星定调「基于别人仓找缺点与我们创新点」；仓寻 Top3；月星拍板主底座继续 MyOS |

**阅读约定**：【已知事实】可在 URL / 仓内文件核对；【推测/建议】不得当协议条款，也不得夸大为已实证算子优势。标 **【已拍板】** 的结论经月星确认，后续工程默认服从。

**材料来源**：三仓公开 README / `Cargo.toml` / 源码；本仓已采纳 novelty / BF16 策略文。仓寻口头 Top3 作候选名单，无独立成文笔记纳入本文件。

---

## 0. 执行摘要

| 段 | 压缩结论 |
|----|----------|
| **A 硬短板** | **ember**：`std`+`rayon`+整表 BF16→f32，无协议 exact / 无 unikernel 分层。**candle**：框架级依赖闭包，非 no_std，不可裁进 Unikernel。**vitni-tensor**：no_std/bit-identical 强，但 **GGUF 主路径**、无 SafeTensors BF16 主赛道与本仓协议指纹契约。 |
| **B 可声明切口** | 相对 ember（对照矿）可辩护的差分钉在：①协议竞技场先行；②no_std core × host 分层；③BF16 字节视图 + 算子内 f32 累加 + `acc_path` 指纹；④对 llama.cpp 的 exact-only 门闩；⑤双赛道分表（ST 主 / Q4 副后置）。勿宣称「手写算子本身更新」。 |
| **C 主底座决议** | **【已拍板】主底座继续 MyOS，不整仓 fork ember。** ember=算子/前向对照矿；candle=行为金标准；vitni=确定性/归约纪律参考（慎整仓）。 |

---

## 1. 对照轴（本项目要什么）

相对「AI-Native Unikernel / 张量推理 OS」+ Phase1 冻结协议，评估维度（**许可外**）：

| 维度 | MyOS / 协议已钉口径 |
|------|---------------------|
| `std` / 依赖 | core：`no_std`+`alloc` 姿态；host runtime 可 std；依赖白名单可控（见 novelty） |
| 确定性 | Phase1 **exact-only** vs llama.cpp；`deterministic=ON`；累计路径进指纹 |
| ST / BF16 | 主赛道 SafeTensors **BF16**；**权重保持字节视图 + 算子内升精**（`acc_path=bf16_w_fp32_acc`） |
| 协议 exact | S1 greedy exact；指纹 `run_fingerprint.json`；CORRECT 不过则吞吐作废 |
| 可裁进 unikernel | core 可脱离 host I/O；tokenizer 可预计算 token 文件（A3） |
| 模型 | `Qwen2.5-1.5B-Instruct`；RoPE=NeoX half；SwiGLU=`down(SiLU(gate)⊙up)` 无 bias；QKV 有 bias |

---

## 2. A · 三仓硬短板（相对上表）

### 2.1 ember — https://github.com/codewithfourtix/ember

| 项 | 【已知事实】 | 相对本项目硬短板 |
|----|--------------|------------------|
| 许可 / 语言 | MIT（`Cargo.toml` `license = "MIT"`；API `license.spdx_id=MIT`）；Rust | 许可可用；**不是**短板主因 |
| 定位 | README：手写 transformer、CPU、无 candle/burn/tch；Qwen2.5；~1300 行 | 用户态推理引擎，**非** OS/Unikernel 叙事 |
| Qwen / ST | Quickstart：`Qwen/Qwen2.5-0.5B-Instruct` + `model.safetensors`；`safetensors` 依赖 | 有 ST 路径；默认真跑 **0.5B**，非协议默认 **1.5B**（可扩，但非现成对齐） |
| BF16 | `src/model.rs` `load_f32`：BF16/F16/F32 **一律转 `Vec<f32>`**；README 模块表写「safetensors loading (bf16→f32)」 | **硬短板**：与本仓「BF16 视图 + 算子内升精、禁止整表物化 f32 主路」**直接冲突**；`M_peak` 叙事与零拷贝加载目标被冲掉 |
| `std` / 依赖 | `std::fs`/`memmap2`/`rayon`/`clap`/`anyhow`/`half`/`serde*`/`tokenizers`（`onig`，需 C 工具链） | **硬短板**：无 `no_std` 分层；`tokenizers`+C 依赖抬高 unikernel / 静态最小环成本 |
| 确定性 | `src/tensor.rs`：`y.par_iter_mut()`（rayon 并行按行）；行内串行点积。验证口径：NumPy oracle「token-for-token」、perplexity、CI live bench——**不是** vs llama.cpp exact 协议 | **硬短板**：并行行归约在跨实现 exact 时通常需钉死线程/归约契约；无 `run_fingerprint` / 赛道分表 / CORRECT 门闩 |
| 量化 | 自研 INT8 / group-INT4；非 GGUF **Q4_K_M** | 副赛道与协议 A1/A2 **不对齐**；易成第二套债 |
| RoPE / SwiGLU | `ops.rs`：明确 HF rotate-half；SwiGLU=`silu(gate)⊙up`；`model.rs`：QKV **有 bias**、MLP 无 bias、tied embed LM head | 算子约定与本仓 **相容**（可作对照矿，不是短板） |
| Unikernel | 无相关设计；CLI host 进程 | **硬短板**：不可「原样裁进」unikernel |

### 2.2 candle — https://github.com/huggingface/candle

| 项 | 【已知事实】 | 相对本项目硬短板 |
|----|--------------|------------------|
| 许可 | Apache-2.0（仓库 license；另有 MIT 徽章/双许可常见于 HF） | 许可可用 |
| 定位 | Minimalist **ML framework**；GPU/CPU/WASM；大量 examples | 框架，不是薄推理 OS |
| Qwen / ST | `candle-examples/examples/qwen`：含 `2-1.5b`；`VarBuilder::from_mmaped_safetensors`；另有 `quantized-qwen2-instruct` GGUF | **金标准对照合适**；作底座则过重 |
| BF16 | 例程：CUDA/Metal 常用 `DType::BF16`，CPU 例常落到 `F32` | 框架 dtype 丰富，但 **无**本仓协议式 `acc_path` 契约 |
| `std` / 依赖 | `candle-core/Cargo.toml`：`gemm`/`rayon`/`safetensors`/`half`/`memmap2`/`rand`/可选 CUDA·Metal·MKL… | **硬短板**：依赖闭包与 no_std unikernel **不相容** |
| 确定性 | 性能/易用取向；非 bit-identical 产品主张 | **硬短板**：不能当 Phase1 exact-vs-llama.cpp 的现成门闩实现 |
| Unikernel | serverless 轻二进制叙事 ≠ unikernel OS | **硬短板**：不可裁进本项目目标形态而不做剥皮重写 |

### 2.3 vitni-tensor — https://github.com/vitnify/vitni-tensor

| 项 | 【已知事实】 | 相对本项目硬短板 |
|----|--------------|------------------|
| 许可 / 商标 | Apache-2.0；README：fork 可用代码，**不可用「vitnify」等商标名** | 许可可用；命名/对外叙事须自有品牌 |
| 定位 | Deterministic、`no_std`、cross-vendor **bit-identical**；`vitnify-receipt v2` | 与「可复现」部分同向，但是 **可验证收据产品**，非张量 OS |
| 权重格式 | README / 复现脚本：**GGUF**（TinyLlama Q4_K_M 等）；量化核 F32·Q4_0·Q4_K·… | **硬短板相对主赛道**：非 SafeTensors BF16 主路；与 Phase1「ST 主 / Q4 副分表、Q4 后置」节奏不合若整仓站其上 |
| `std` | `#![no_std]` + `alloc`；deps：`libm`、`blake3`（`default-features=false`）；可选 `std-parallel`+rayon（声称按行独立故 bit-identical） | no_std **强项**；但仍是引擎 crate，不是 WeightLoader/ModelCore/协议竞技场 |
| 确定性 | 钉死归约顺序 + regime；跨 CPU/GPU 同 digest（regime-2） | 主张强于 ember/candle；**但是** digest/regime **≠** 本仓 vs **llama.cpp greedy exact** 协议（不同参考系） |
| Qwen | README：测过 Qwen2.5；RoPE 支持 NeoX | 有帮助；仍缺本仓布局钉死 + ST BF16 视图策略 |
| Unikernel | 面向 host/GPU backends + receipt；非 unikernel 页表/最小环 | **部分可借鉴**；不可原样当 AI-Native OS 底座 |

---

## 3. B · 相对 ember 可声明的创新切口（非整仓 fork）

前提：切口必须相对 ember **可证伪差分**，并对齐已采纳 novelty（协议先行 / no_std 分层 / 双赛道 exact）。**【已拍板】** 主底座是 MyOS，下列切口是「站在 MyOS 上相对对照矿可讲什么」，**不是**「fork ember 后再改什么」。**禁止**把「手写 RMSNorm/RoPE/GQA/SwiGLU」当创新（ember 已有且 RoPE/SwiGLU 已对齐）。

| # | 切口 | 相对 ember 的可核对差分 | 对齐本仓口径 |
|---|------|-------------------------|--------------|
| 1 | **对打协议先于算子完成** | ember：perplexity / NumPy oracle / CI bench；**无**冻结场景 S1–S6、指纹、±5%、CORRECT 一票否决 | `PHASE1-ARENA-PROTOCOL-v0.1.md`；novelty「协议先行」 |
| 2 | **no_std core × host runtime 分层** | ember：单体 `std` bin + rayon + tokenizers | MyOS crates：`weight_loader`/`tensor_core`/`model_core` no_std 姿态；runtime host |
| 3 | **BF16 字节视图 + 算子内 f32 累加 + `acc_path` 指纹** | ember：加载期整表 `bf16→f32` | `phase1-bf16-compute-strategy.md`；`acc_path=bf16_w_fp32_acc` |
| 4 | **exact-only vs llama.cpp（主基线）** | ember：自研 oracle / 质量表；未钉 llama.cpp CPU 同指纹对打 | 协议 F2/A7/§4.1 |
| 5 | **双赛道分表：ST-BF16 主胜负 / GGUF-Q4_K_M 副表后置** | ember：自定义 INT8/INT4 与 f32 混在同一产品叙事 | 协议 A1/A2；novelty「双赛道 exact」 |

**【推测/建议】对外话术红线**：在 S1 exact 通之前，不得宣称吞吐/内存优于 llama.cpp 或 ember；不得宣称 Unikernel「已落地」。

---

## 4. C · 主底座决议

### 4.1 结论（钉死）

**【已拍板】主底座继续 MyOS；不以 ember 为 Git 主底座整仓 fork 替换。**

支撑理由（事实驱动，拍板前分析仍成立）：

1. ember 的 **整表转 f32** 与已采纳 BF16 主路 **相反**；fork 后若立刻改回视图升精，等于重写加载与热路径，**吃不到「现成底座」红利**。
2. ember 的 **rayon + tokenizers/onig + std 单体** 与 unikernel / no_std 分层目标冲突；剥依赖成本接近自研分层。
3. MyOS **已有**：冻结协议、布局钉、SafeTensors mmap、BF16 路径与 novelty 文档——继续 MyOS 的目标一致性更高。
4. ember 公开活跃度与工程成熟度不足以压过协议/分层已投入（API：`stargazers_count` 低、推送窗口短——以查阅当时为准）。

### 4.2 三仓角色（拍板后默认）

| 角色 | 决议 | 用法 |
|------|------|------|
| **主底座** | **MyOS（本仓）** | 唯一合入面；GitHub 写操作仅 OSdeveloper |
| **算子/前向对照矿** | **ember** | 对照 Qwen2 ST 前向结构、bias 规则、RoPE half、SwiGLU；**移植思想，不整仓替换** |
| **金标准行为对照** | **candle** `qwen` / `quantized-qwen2-*` | 核对 HF 布局与生成行为；**禁止**引入 candle 为 core 依赖 |
| **确定性/归约纪律参考** | **vitni-tensor**（文档级） | 借鉴「钉死 reduction / regime」写法；**不**把 receipt 产品叙事或 GGUF-first 整仓迁入；注意商标 |

### 4.3 可接受的窄例外（仍非「站 ember」）

**【推测/建议】** 若仅为加速「第一次连贯 decode」做 **抛掷型 spike**：可临时跑 ember 可执行文件做行为对照；**成果只允许以算子差分/测试向量形式合入 MyOS**，不改变默认分支血统为 ember fork。

---

## 5. 材料索引（可核对）

| 源 | URL / 路径 |
|----|------------|
| ember README / Cargo / ops / model / tensor | https://github.com/codewithfourtix/ember |
| candle README / candle-core / qwen example | https://github.com/huggingface/candle |
| vitni README / Cargo / regimes | https://github.com/vitnify/vitni-tensor |
| 本仓协议与口径 | `PHASE1-ARENA-PROTOCOL-v0.1.md`；`docs/phase1-novelty-risks.md`；`docs/phase1-bf16-compute-strategy.md`；`README.md` |

---

## 6. 变更记录

| 日期（UTC+8） | 说明 |
|---------------|------|
| 2026-11-03 | 初稿：Top3 硬短板 + 相对 ember 切口 + 主底座建议（深析） |
| 2026-11-03 | 合入整理：月星拍板「主底座继续 MyOS / 不整仓 fork ember」写入正文；章节用语从「若 fork」改为「相对对照矿可声明切口」（深析） |
