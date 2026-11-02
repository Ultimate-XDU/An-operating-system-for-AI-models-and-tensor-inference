# Phase1 创新点与风险分析（WeightLoaderCore / ModelCore / 最小推理循环）

| 项 | 内容 |
|----|------|
| 文档 | `docs/phase1-novelty-risks.md` |
| 状态 | 深析 Phase1 起步分析（论证附录候选） |
| 日期 | 2026-11-02（UTC+8） |
| 效力 | **分析意见**，不修改冻结协议；协议冲突以 `PHASE1-ARENA-PROTOCOL-v0.1.md` 为准 |
| 范围 | 仅本仓 `/workspace/MyOS`；不扩及其他项目 |

**阅读约定**：凡标 **【已知事实】** 均可在协议/代码/README 中核对；凡标 **【推测/建议】** 为论证或工程建议，**不得当作协议条款**。

---

## 0. 执行摘要（给 OSdeveloper 同步）

1. **现状**：Phase1 评测框架已冻结（v0.1）；实现仍为 stub（`WeightMap` / `MinimalistModel` trait、`gemv_f32`、runtime println）。workspace crates **当前无外部 crate 依赖**（`Cargo.lock` 可核）。
2. **可辩护新颖点（收窄后）**：不是「又一个 LLM runtime」，而是（a）**对打协议先于算子**的可复现竞技场；（b）**core crates 的 no_std + 零外部依赖姿态**为后续 Unikernel/最小环留接口债可控的底座；（c）**双赛道分表 + exact 前置**把正确性与吞吐叙事拆开。相对 llama.cpp / candle / burn / tch，Phase1 **尚未**有算子/内核层实证优势。
3. **最大风险**：exact 门闩在 BF16/FP16 累计路径未钉死时易 **假阴性**；Q4_K_M 副赛道易变成第二套实现债；零依赖会推迟 tokenizer/SIMD/加速库，S1 exact 对齐可能极慢；仓库内 **无成文「四阶段」总路线文档**，与后续衔接需先书面钉死 Phase2+ 边界，否则接口债隐性累积。
4. **建议立刻可执行**：按下文 §6 的门闩清单落地 `run_fingerprint` / exact 对齐 S1 / 赛道分表 / 累计路径写入指纹；**主宣称只走 SafeTensors BF16或FP16 一档**，Q4_K_M 副赛道可延后到 S1 exact 通后再开。

---

## 1. 材料与差分基线

### 1.1 【已知事实】已读材料

| 材料 | 要点 |
|------|------|
| `PHASE1-ARENA-PROTOCOL-v0.1.md` | 冻结 F1–F5；范围 = WeightLoaderCore + ModelCore + 最小推理循环（prefill+decode）；主基线 llama.cpp；CPU-only；exact 门闩；场景 S1–S6 + S3+ 附录 |
| `PHASE1-ARENA-PROTOCOL-v0.md` | 历史草案；开放问题在 v0.1 已关闭（§7） |
| `README.md` | AI-Native Runtime / 极简 Unikernel；Phase1 四点 focus；GitHub 写操作仅 OSdeveloper |
| `Cargo.toml` + 四 crate | workspace：`weight_loader`, `tensor_core`, `model_core`, `runtime` |
| `Cargo.lock` | 仅上述四包，**无第三方依赖条目** |
| `crates/*/src` | stub：`WeightMap`、`gemv_f32`、`MinimalistModel`、`main` 打印占位 |
| `docs/`、`scripts/` | 目录存在但（写本分析前）无其他内容 |

### 1.2 【已知事实】v0 → v0.1 对分析有影响的差分

- **F1**：钉死 `Qwen2.5-1.5B-Instruct`；主赛道 SafeTensors **BF16/FP16**；副赛道 GGUF **Q4_K_M**；回退 0.5B 须全表重跑并标注（A9）。
- **F2**：正确性 **只认 exact**；logits/token tol → Phase2。
- **F3**：S3 正式 `n_prompt=4096`；8k → **S3+ 附录**（不进主胜负）。
- **F4**：Phase1 **只打 CPU**；GPU → Phase2 附录。
- **F5**：±5% 胜负带宽维持。
- 指纹字段补 `model_id`、`quant_or_dtype`，`device=cpu` 写死进示例。

### 1.3 【推测/建议】关于「四阶段总路线」

仓库内 **未发现** 名为「四阶段」或 Phase3/Phase4 的冻结文档。协议与 README **明确出现**的仅有：

- Phase1：加载 + 模型抽象 + CPU 最小环 + 对打（协议范围句、README Phase 1 focus）。
- Phase2（协议点名）：logits tol；GPU/CUDA 附录（F2、F4、A8、§4.1 附录对照）。

因此下文「四阶段衔接」中的 Phase3/Phase4 划分均为 **建议性叙事骨架**，供 OSdeveloper/博弈手后续成文；**不是**当前协议事实。

---

## 2. 零依赖基线如何站住

### 2.1 【已知事实】当前「零依赖」证据

| 证据 | 内容 |
|------|------|
| 各 crate `Cargo.toml` | `weight_loader` / `tensor_core` 的 `[dependencies]` 为空；`model_core` / `runtime` 仅 path 依赖 workspace 兄弟 crate |
| `Cargo.lock` | 无 `serde` / `memmap2` / `anyhow` / `half` / `candle-*` / `tch` 等第三方包 |
| `#![cfg_attr(not(test), no_std)]` | 出现在 `weight_loader`、`tensor_core`、`model_core` 的 `lib.rs`（非 test 构建） |
| `extern crate alloc` | `weight_loader`、`model_core` 使用 alloc，说明允许堆，但非 std-only API 绑定 |
| `runtime` | **有 std**（`println!` bin）；描述为 “Minimal **host-side** inference loop” |

结论（事实层）：Phase1 **core 三库**目前保持「无外部 crate + no_std 姿态」；**host runtime bin 可以依赖 std**。这与 README「极简 Unikernel」方向相容，但 **Unikernel 本身尚未在代码中出现**。

### 2.2 【已知事实】协议对「零依赖」的直接表述

协议 **没有** 写「禁止任何第三方 crate」或「必须零依赖」。协议允许的差异（§1.1）包括：Unikernel / 用户态最小环 vs llama.cpp 进程；分配器、页表、静态链接；内部算子实现（仍须服从 §2.4 数值协议）。

因此：**零依赖是实现策略与接口债管理选择，不是对打公平性条款。** 引入依赖只要不破坏指纹对齐与 deterministic 要求，不自动犯规。

### 2.3 相对常见路径：什么真能辩护、什么不能

| 路径 | 典型依赖/形态 | Phase1 相对差异（可辩护？） |
|------|----------------|------------------------------|
| **llama.cpp** | C/C++、丰富量化与 CPU/GPU 后端、成熟 tokenizer | 协议已选定为 **主基线与 exact 参考**（§4.1）。差异不在「更能打」，而在 MyOS 要证明 **更薄的运行时形态**仍能 CORRECT + 可复现指标（A7）。**实证优势尚未存在**（实现为 stub）。 |
| **candle** | Rust、HuggingFace 生态、GPU/CPU、大量 crate | MyOS 当前无 candle 式框架依赖；可辩护点是 **更小的依赖闭包 / no_std 核心**，不是训练或模型覆盖面。 |
| **burn** | 后端可插拔、机器学习框架取向 | 同上；Burn 优化的是框架通用性，MyOS Phase1 优化的是 **对打可复现的最小推理环**。 |
| **tch / libtorch** | 绑定 libtorch，重依赖 | MyOS 明确避开此路径（现状零第三方）；可辩护为部署面与审计面更小，**代价是算子与数值工具链全自建**。 |
| **llama2.c / llama2.rs** | 协议已列为附录对照（§4.1） | 教育级最小实现；MyOS 若只做到同级 stub，**新颖性不足**；必须靠协议竞技场 + 双格式加载 + 后续 Unikernel 形态拉开。 |

**【推测/建议】可辩护的 Phase1「新颖点」收窄为三条（避免夸大）：**

1. **评测契约先行**：在算子完成前冻结场景、指纹、exact 门闩与胜负合成（v0.1），使后续优化不能靠改规则 retrospectively 宣称胜利——这是 **方法/过程创新**，不是算子创新。
2. **格式无关加载面 + 模型 trait 面**：`WeightMap` / `MinimalistModel` 把「文件字节视图」与「逐 token forward」切开（代码级意图；实现未完成）。相对「一个巨石 `llama_eval`」，有利于双赛道分表（A1）与后续换后端。
3. **no_std 核心 × host 环分层**：core 保 no_std，runtime 先做 host 对打——为 Unikernel 迁移留缝。这是 **架构赌注**，Phase1 只能证明「分层没挡住 S1 exact」，不能证明 Unikernel 已赢。

**【推测/建议】不应对外宣称的点（当前无证据）：**

- 「已快于 / 省内存于 llama.cpp」（无 `runs/`、无 cost card）。
- 「零拷贝 SafeTensors/GGUF 已完成」（README 写目标；`WeightMap` 仅为 placeholder）。
- 「AI-Native OS / Unikernel 已落地」（README 愿景；代码仍为 host stub）。

### 2.4 零依赖基线「站住」的操作定义（建议）

**【推测/建议】** 建议 OSdeveloper 把「零依赖基线站住」写成可验收的工程门闩（协议未写死，属新增）：

| 级别 | 含义 | 验收 |
|------|------|------|
| Z0 | 现状 | `Cargo.lock` 无第三方；`cargo build -p myos_runtime` 通过 |
| Z1 | 能加载 | SafeTensors **或** GGUF 之一：只读映射 + 按名取 tensor 字节；仍可无第三方（手写解析）或 **显式白名单** 单 crate（若引入须记入文档「依赖例外」） |
| Z2 | 能对齐 | 主赛道上 S1 greedy exact vs llama.cpp PASS（协议 E2） |
| Z3 | 可复现对打 | S1/S2 产出 `run_fingerprint.json` + `tokens.txt` + cost card（协议 §6） |

**零依赖会丢掉什么（风险，建议层）：**

- 成熟 tokenizer / chat template → 必须用 **预计算 token 序列文件**（协议 A3 已允许），否则自研 tokenizer 易成 CORRECT-FAIL 来源。
- SIMD / BLAS / Accelerate → 吞吐可能长期落后；协议允许算子自研，但 ±5% 带宽下「慢但正确」仍可能 Phase1 负（§4.3）。
- `half` / 硬件 BF16 支持 → 需自管 BF16↔FP32 转换与累计路径；与 exact 门闩强耦合（见 §3）。
- 量化 kernel（Q4_K_M）→ 副赛道几乎必然大量代码；威胁「薄运行时」叙事。
- 测试与基准脚手架 → `scripts/` 空；若坚持零依赖，应用最小 shell + 双方约定的 JSON schema，而非再引入重型 bench 框架。

**【推测/建议】** Phase1 **允许 runtime（host）引入极窄白名单**（如仅用于指纹 JSON 序列化），但 **weight_loader / tensor_core / model_core 继续零第三方**，以保护 Unikernel 迁移面。白名单变更应记 `docs/dependency-exceptions.md`（尚不存在，建议创建时机由 OSdeveloper 定）。

---

## 3. BF16/FP16 与 Q4_K_M 双赛道的数值门闩策略

### 3.1 【已知事实】协议钉死的门闩

| 条款 | 内容 |
|------|------|
| F1 / A1 / A2 | 主：SafeTensors BF16 **或** FP16（双方一致）；副：GGUF Q4_K_M；**分表**；跨档禁止直接宣称胜负 |
| F2 / §2.4 | Phase1 **只认 exact match**（greedy + deterministic=ON）；与 llama.cpp 同权重同赛道逐 token；logits tol → Phase2 |
| §2.3 | deterministic=ON：R 次序列逐 token 一致，否则 DET-FAIL |
| A4 | 主表默认 deterministic=ON；固定 seed；禁非确定 kernel |
| A7 | 正确性不过关 → 吞吐胜负作废 |
| §1.2 / 指纹 | 「权重精度 + **累计路径**」写入配置哈希；指纹不同禁止并表 |
| Cost card | `track: SafeTensors BF16/FP16 \| GGUF Q4_K_M`；greedy exact / det 分项 |

### 3.2 双赛道各自的数值含义（事实 + 建议）

**主赛道（SafeTensors BF16/FP16）**

- 【已知事实】双方须同一 dtype 档；exact 相对 llama.cpp。
- 【推测/建议】BF16 与 FP16 **不要在主表混用**；F1 写「BF16/FP16」是可选一档，不是「可混报」。建议 scoreboard 列 `quant_or_dtype=bf16` **或** `fp16`，选一为主声明档。
- 【推测/建议】「累计路径」应至少区分：`acc=fp32`（权重 BF16/FP16、matmul 累加 FP32）vs 纯低精度累加。llama.cpp 默认路径需在基线 cost card 写明，否则 exact 失败无法归因。

**副赛道（GGUF Q4_K_M）**

- 【已知事实】与主赛道分表；不能拿 Q4 胜去宣称打败主赛道上的对手。
- 【推测/建议】Q4_K_M 的 dequant + 累加顺序对 greedy token 极敏感；exact 对齐 llama.cpp 通常要求 **同 block 布局、同 dequant 阶、同 reduce 顺序**。这是高维护成本赛道。
- 【推测/建议】副赛道 Phase1 的战略价值是：（1）内存墙（S3）对照；（2）证明 WeightLoader 格式面真实存在。**不建议**用副赛道冲击「Phase1 宣称胜」（§4.3 主胜负未要求双赛道都胜）。

### 3.3 Exact 门闩的假阳性 / 假阴性

| 类型 | 含义 | Phase1 场景下的典型原因 | 缓解（建议） |
|------|------|-------------------------|--------------|
| **假阴性**（实现其实「够好」但 CORRECT-FAIL） | 正确性门闩过严导致吞吐分被作废 | 累加顺序/FMA 差异；BF16 转换点不同；RoPE/RMSNorm 常数；prefill 与 decode 数值路径不一致；线程归约顺序（即使 seed 固定） | 钉死单线程参考跑一次；指纹写入 `n_threads` 与 `acc_path`；先 S1 `n_gen` 缩短做差分定位；**不**在 Phase1 引入 tol（违反 F2） |
| **假阳性**（exact PASS 但语义/实现有洞） | 门闩通过但结论外推过度 | 只对单一 prompt / 短 `n_gen` 对齐；采样器未测（greedy-only）；KV 在长上下文翻车但 S1 通过 | 协议已要求场景矩阵；建议 **S1 PASS 不得外推 S3**；DET 一致性 R=5 必须跑满（复述 §2.3） |
| **赛道串味** | 用错权重或跨档宣称 | SafeTensors vs GGUF 混表；BF16 vs Q4 比 tokens/s | 指纹校验自动化（§1.3）；scoreboard 分轨（复述 A1） |

**【已知事实】** Phase1 **拒绝** logits tol（F2），因此「假阴性偏多」是 **有意的方法选择**：先换可复现对齐，再在 Phase2 放开数值邻域。论证时应主动承认该偏置，而非事后抱怨门闩过严。

### 3.4 双赛道维护成本与接口债

**【推测/建议】**

| 成本项 | 主赛道 BF16/FP16 | 副赛道 Q4_K_M |
|--------|------------------|---------------|
| 解析 | SafeTensors header + dtype | GGUF 计算图元数据 + quant block |
| Kernel | GEMV/GEMM、RMSNorm、RoPE、Attn、SiLU | 额外 dequant 或 fused qGEMV |
| Exact 对齐 | 中高（累加路径） | 高（须贴近 llama.cpp quant 语义） |
| 对 Unikernel | 映射文件 + 页锁定较直观 | 同左，但代码体积与测试倍增 |

**建议策略（新增，非协议原文）：**

1. **P1-A（主路）**：只打通 SafeTensors **单一 dtype**（建议先 **FP16 权重 + FP32 accumulate**，因工具链与调试更常见；若宿主机/基线以 BF16 为准再锁 BF16）。
2. **P1-B（副路）**：WeightLoader 能 **识别并映射** Q4_K_M tensor 字节即可先记「加载对抗 S6」；**forward exact** 可标 N/S 或延后，只要 scoreboard 不混报。
3. 禁止在 ModelCore 里用 `#[cfg(track)]` 分叉出两套模型逻辑长期共存；量化差异应落在 **tensor_core 的后端 trait** 或 weight 视图类型上，避免 Phase2 GPU 时爆发式分叉。

---

## 4. 与「四阶段」总路线的衔接

### 4.1 【已知事实】协议内已写的阶段边界

| 阶段信号 | 协议位置 | 内容 |
|----------|----------|------|
| Phase1 范围 | 文首、README | WeightLoaderCore + ModelCore + 最小推理循环；CPU；对打 llama.cpp |
| Phase2 | F2、F4、A8、§4.1 | logits/token tol；GPU/CUDA 附录；vLLM 仍非主胜负 |
| 代价卡反馈环 | §5 结论句 | 「是否值得为 Phase2 继续加码」 |
| 实验步进 | §8 E0–E3 | 冻结 → 基线 cost card → MyOS S1 exact → 协议修订 |

### 4.2 【推测/建议】建议成文的四阶段骨架（供路线图文档，非正式协议）

> 下列命名供 `docs/roadmap.md`（建议路径）讨论；**未写入则不得当事实引用。**

| 阶段 | 目标（建议） | 与 Phase1 产出的衔接点 | 接口债风险 |
|------|--------------|------------------------|------------|
| **Phase1** | Host 最小环 + 协议竞技场 + exact | `WeightMap`、`MinimalistModel`、指纹与 tokens 产物 | trait 过于窄（仅 `forward_token`）可能不够表达 prefill 整段 |
| **Phase2** | GPU 附录 + 数值 tol + 更强算子 | 累计路径/设备字段已在指纹预留；`device=cpu` 需泛化 | 若 core 泄漏 host SIMD 内禀，GPU 需重写；logits tol 需单独门闩文档 |
| **Phase3** | 调度 / 批 / 多请求（S4/S5 从 N/S 转正） | 协议已预留 B>1 场景 | `MinimalistModel` 无 batch 维；KV 布局需一开始避免写死「单请求」 |
| **Phase4** | Unikernel / 极简 OS 形态落地 | README 愿景；§1.1 允许形态差异；M_peak 已允许 Unikernel 页占用口径 | runtime 若深度绑定 Linux RSS/`println` 工具链，迁移成本高；加载路径需早用「可换 mapper」 |

### 4.3 Phase1 → 后续的具体接口债清单（建议跟踪）

1. **`MinimalistModel::forward_token` 只覆盖 decode 步**：【事实】签名在 `model_core/src/lib.rs`；【建议】补 `prefill(&[u32], &mut Kv)` 或允许 `pos` 段式更新，否则协议中的 prefill/decode 计时边界（§2.1）难在 API 上强制。
2. **`WeightMap::tensor_bytes` 无 dtype/shape**：【事实】仅返回 `&[u8]`；【建议】返回结构化 `TensorView { dtype, shape, strides, bytes }`，否则 ModelCore 只能硬编码 Qwen2.5-1.5B 布局，双赛道与回退 0.5B（A9）会复制粘贴。
3. **`gemv_f32` 唯一算子**：【事实】`tensor_core` 仅此；注释提到 RMSNorm/Softmax stubs 但未实现；【建议】先补齐 Qwen2 所需最小 op 集表（文档化），再谈优化。
4. **指纹与 runs 目录未落地**：【事实】协议 §6 要求；仓库无 `runs/`、`arena/`；【建议】与实现同步建目录约定，避免事后无法宣称可复现。
5. **Tokenizer 策略**：【事实】A3 允许共享 token 文件；【建议】Phase1 固定「外置 tokens 文件」为唯一路径，把 tokenizer 整段推到 Phase2+ 或附录，保护零依赖核心。

---

## 5. 可发表 / 对外叙事：Claim 与证据门槛

**【推测/建议】** 若 Phase1 要写成论文/技术报告章节，建议只用下列 **弱而可证** 的 claim；括号内为证据门槛。

| ID | Claim（建议表述） | 证据门槛 | 当前状态 |
|----|-------------------|----------|----------|
| C1 | 我们给出一套 CPU 上、与 llama.cpp 公平对打的 **可复现协议**（场景、指纹、exact、±5% 合成） | 公开 v0.1 文本 + 至少一方完整 cost card 流水线 | 协议 ✅；流水线 ❌ |
| C2 | 在 **相同权重与 deterministic greedy** 下，挑战者可与参考实现 **逐 token exact 对齐**（S1） | `tokens.txt` 一致 + fingerprint 对齐 + DET R 次一致 | 实现 stub ❌ |
| C3 | 核心推理库可在 **无第三方 crate / no_std** 约束下完成主赛道一次完整 generate | `Cargo.lock` 审计 + S1 功能跑通 | 依赖姿态 ✅；功能 ❌ |
| C4 | 双格式加载面（SafeTensors 与 GGUF）可分表评测且拒绝跨档混报 | 两赛道各至少一张合法 scoreboard；混报检测脚本 | 意图 ✅；实现 ❌ |
| C5（慎用） | 在 CORRECT 前提下，于 S1/S2 相对 llama.cpp 取得协议定义之「场景胜」 | §4.2–4.3 全表 | 无数据 ❌ |

**叙事结构建议：** Methods = 协议 + 架构分层；Results = exact 对齐与 scoreboard；Discussion = 零依赖代价、exact 假阴性、Q4 维护成本、通向 GPU/Unikernel 的接口债。**不要**把 README 愿景句当作 Results。

---

## 6. Phase1 门闩 / 验收建议（3–7 条）

| # | 建议 | 复述 or 新增 | 对齐 |
|---|------|--------------|------|
| G1 | 每次计分跑必须产出 `run_fingerprint.json`，字段含协议 §1.3 列表；缺指纹不得入表 | **复述** | §1.3、§6 |
| G2 | 主表默认 deterministic=ON；同指纹 R=5 greedy `tokens.txt` 逐 token 自洽，否则 DET-FAIL | **复述** | §2.3、A4 |
| G3 | 与 llama.cpp 同赛道同权重：S1 greedy **exact** PASS 前，禁止用吞吐对外宣称胜负 | **复述**（强化 E2） | §2.4、A7、§8 E2 |
| G4 | SafeTensors 主赛道与 GGUF Q4_K_M 副赛道 **分文件/分表**；CI 或脚本拒绝跨 `fmt`/`quant_or_dtype` 合并胜负行 | **复述** + 自动化属 **新增** | A1、A2、F1 |
| G5 | 指纹增加并文档化 `acc_path`（如 `fp32_acc`）与主声明 dtype；BF16 与 FP16 不得无说明混入同一主表 | **新增**（协议仅写「累计路径写入配置哈希」，未钉枚举名） | §1.2 精度行 |
| G6 | Phase1 核心三库保持无第三方依赖；若 host runtime 引入白名单 crate，须在 `docs/dependency-exceptions.md` 记录理由与卸载条件 | **新增** | 代码现状 + README Unikernel 方向 |
| G7 | S4/S5 在 batch 未实现时标 N/S；S3+/S6 不得单独支撑「打败 llama.cpp」对外句；总胜负按 §4.3 | **复述** | §3、§4.3 |

---

## 7. 风险总表（浓缩）

| 风险 | 严重度 | 说明 | 缓解 |
|------|--------|------|------|
| Exact 假阴性阻断对打 | 高 | F2 故意偏严；累加路径未钉死时 S1 长期 FAIL | G5；单线程参考；共享 token 文件（A3） |
| 双赛道实现分叉 | 高 | Q4 与 BF16/FP16 两套 kernel | P1-A 先行；量化差异限在 tensor 后端 |
| 零依赖导致进度过慢 | 中高 | 自研解析+算子+无 SIMD | Z1/Z2 门闩；tokenizer 外置；必要时 runtime 白名单 |
| `forward_token` API 过窄 | 中 | prefill 计时与批处理难表达 | 阶段衔接时扩展 trait（§4.3） |
| 四阶段未成文 | 中 | Phase3/4 口头化导致暗债 | 补 `docs/roadmap.md`（建议） |
| 过早对外「OS/Unikernel」宣称 | 中 | 与 stub 现实落差 | 对外只用 C1–C2 类 claim |
| 回退 0.5B 被误用为胜 | 低 | A9 要求标注与全表重跑 | scoreboard 强制 `MODEL_FALLBACK` |

---

## 8. 结论

Phase1 当前 **真正硬的资产**是已冻结的对打协议与清晰的 crate 分层意图；**真正软的资产**是尚未实现的加载、算子与 exact 对齐。零依赖 / no_std 核心是可辩护的架构姿态，但只能支撑到「薄、可迁移」叙事，不能替代 S1 exact。双赛道中应 **主赛道换 exact、副赛道先证明可加载**，避免 Q4_K_M 拖垮 Phase1 工期。与后续阶段的衔接取决于尽早把 `TensorView`、prefill API、设备/累计路径字段和路线图文档写清楚——否则 Phase2 GPU 与 Unikernel 会偿还隐形接口债。

---

*起草：深析工作代理 · 仅服务本仓 · 不合入 git（交 OSdeveloper）*
