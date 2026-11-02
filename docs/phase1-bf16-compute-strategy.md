# Phase1 BF16 计算策略短结论（深析 → OSdeveloper）

| 项 | 内容 |
|----|------|
| 文档 | `docs/phase1-bf16-compute-strategy.md` |
| 状态 | 分析意见；**不改代码**；不修改冻结协议 |
| 日期 | 2026-11-02（UTC+8） |
| 效力 | 冲突时以 `PHASE1-ARENA-PROTOCOL-v0.1.md` 为准；口径对齐 `docs/phase1-novelty-risks.md` |
| 范围 | 仅 `/workspace/MyOS` |

**阅读约定**：【已知事实】可在协议/布局/代码核对；【推测/建议】不得当协议条款。

**勘误（2026-11-02）**：§2 表中 RoPE 原写「偶奇维交错」不准确；Qwen2.5-1.5B-Instruct 应对齐 **NeoX half-rotate**（已合入 main @ `7db6f713` / PR #9）。相邻 interleaved **不是** Phase1 合法配对。

---

## 0. 材料核对（背景 → 事实）

| 陈述 | 判定 | 依据 |
|------|------|------|
| SafeTensorsWeights + host mmap 已落地 | 【已知事实】 | `weight_loader/src/safe_tensors.rs`、`mmap.rs`；`host` → `memmap2`；runtime `--weights` 只打印名 |
| 布局钉 Qwen2.5-1.5B-Instruct 全 BF16、无 `lm_head`（tied） | 【已知事实】 | `docs/qwen25-1.5b-instruct-layout.md`：338/338=`BF16`，`tie_word_embeddings=true`；`qwen25_layout.rs` 常量 |
| `load_f32_tensor` 仍只吃 F32 | 【已知事实】 | `weight_loader/src/lib.rs`：非 `DType::F32` → `DTypeMismatch`；测例对 F16/BF16 亦拒 |
| 算子侧仅有 `gemv_f32`；真实 Qwen forward 未接 | 【已知事实】 | `tensor_core` 仅 GEMV；`QwenStub` 仍用玩具 `tok_embeddings`/`lm_head` f32 |
| Phase1 只认 exact、只打 CPU；Q4 副赛道分表 | 【已知事实】 | 协议 F2/F4/A1/A2 |
| 主路单一 dtype + `acc_path` + Q4 后置 | 【已知事实】（先前意见已采纳） | `docs/phase1-novelty-risks.md` §3.4 / G5；OSdeveloper 同步 |
| core「零第三方」 | 【已知事实·已变】 | `weight_loader` 现依赖 `safetensors`（默认）+ 可选 `memmap2`；`tensor_core`/`model_core` 仍无外部 crate。与 novelty 文档写稿时「Cargo.lock 无第三方」**不一致**——以当前 `Cargo.toml` 为准 |

---

## 1) 主赛道 S1 greedy exact：权重视图策略

**【推测/建议】主推荐：权重保持 BF16 字节视图 + 算子内升精度（元素级 BF16→f32，累加 f32）。**

| | 内容 |
|--|------|
| 一句代价 | 不能再指望 `load_f32_tensor` 喂满模型；须在 `tensor_core`（或薄适配层）实现 BF16 解码的 GEMV/归约，实现与调试面变宽。 |
| 为何不选「加载时整表转 f32」作主路 | 【已知事实】官方存储与布局钉均为 BF16（约 2.88 GiB）；mmap 路径刻意「payload 不拷入 `Vec`」。【推测/建议】加载期物化 f32 ≈ 再占一份 ~5.76 GiB 量级权重工作区，直接恶化 S3 `M_peak`，并削弱已落地的零拷贝叙事；且与 llama.cpp 同档「BF16 权重文件」指纹更难对齐。 |
| 可接受的窄例外 | 【推测/建议】单测/差分可用「按名临时 decode 一小块到 f32」；**禁止**把整模预转 f32 写成 Phase1 主加载路径。 |
| 与 novelty 口径 | 布局已钉 **BF16 主声明档**（非「先 FP16」）；累计路径仍建议指纹写 `acc_path=bf16_w_fp32_acc`（G5）。 |

---

## 2) Phase1 CPU 最小可行：算子优先序

**【已知事实】** 真实块所需能力来自布局/config：RoPE（`rope_theta=1e6`）、GQA（Q=12/KV=2/head=128）、SwiGLU（`silu`）、RMSNorm（`eps=1e-6`）、QKV **有** bias、`o_proj`/MLP **无** bias、输出 = tied `model.embed_tokens.weight`。

**【推测/建议】** 有序清单（先阻塞 exact 端到端者优先；线性投影视为 attn/MLP/lm_head 的内嵌原语，不另开赛道）：

| 序 | 算子 | 为何卡 exact |
|----|------|----------------|
| 1 | **embed gather** | 一切 token 路径起点；错行/错 dtype 升精 → 全序列必偏 |
| 2 | **RMSNorm** | 每层×2 + final；`Σx²` 归约顺序与 `eps` 是经典 exact 杀手 |
| 3 | **RoPE** | Q/K 旋转；**配对钉死 HF Qwen2 / llama.cpp `LLAMA_ROPE_TYPE_NEOX` half-rotate**（lane `i` 与 `i+head_dim/2`，非相邻 even/odd interleaved）；`theta`、freq 表与 half 切分须与参考逐位同构，否则后续 attn 全漂 |
| 4 | **GQA attn**（含 Q/K/V/O 投影+bias 规则、KV 读写、score、softmax、×V、KV 头复制） | 最长依赖链；softmax 与 matmul 归约顺序、prefill/decode 是否同核，直接决定 token |
| 5 | **SwiGLU**（SiLU(gate)⊙up → down） | `silu` 近零数值与三个 GEMV 的累加顺序；与 attn 并列的第二大漂源 |
| 6 | **tied lm_head** | 用 embed 矩阵做 vocab 维投影；vocab=151936，最大单次 GEMV；greedy 平局规则已在 `greedy_token`（较低 index）须与参考一致 |

**落地暗示（建议，非代码任务）**：先做「BF16 行/矩阵 × f32 向量 → f32」的统一原语，再叠 1–6；S1 exact 通之前不做第二套 dtype 核。

---

## 3) exact 门闩下 BF16 累加顺序：风险与建议

**【已知事实】** F2 = Phase1 **只认 exact**，logits tol → Phase2；§1.2 要求「权重精度 + **累计路径**」进配置哈希；deterministic=ON 禁非确定 kernel。

**【推测/建议】风险**

| 风险 | 说明 |
|------|------|
| 升精时机 | 权重 BF16 保留 vs 中间激活是否写回 BF16；多一次 round-trip 即可改 argmax |
| 归约顺序 | 行内左→右串行 vs 分块/多线程归约；FMA 合乘加 vs 先乘后加 |
| 路径分叉 | prefill 整段 vs decode 逐步若用不同核/不同累加器宽度 → S1 短序列「碰巧过」、长序列炸 |
| 假阴性 | 实现「观感正确」但与 llama.cpp CPU 同赛道路径不一致 → CORRECT-FAIL 且吞吐分作废（A7） |

**【推测/建议】建议（相对 exact-only，不引入 tol）**

1. 钉死并写入指纹：`quant_or_dtype=bf16`，`acc_path=bf16_w_fp32_acc`（名称可微调，但必须稳定枚举）。
2. **激活在层间保持 f32**；仅从权重字节视图读 BF16 并即时升精参与乘加。
3. exact 参考跑：**先单线程**、固定串行归约；再开 `n_threads`，线程数已在指纹中。
4. 基线 llama.cpp 同权重同 CPU 路径的累加方式写进首张 cost card，失败时归因到路径而非「感觉精度不够」。
5. **不**在 Phase1 用 logits tol 掩盖顺序差（违反 F2）。

---

## 4) 明确不做什么（Phase1 边界）

对齐协议 + novelty 已采纳执行口径：

| 不做 | 依据 |
|------|------|
| **GGUF / Q4_K_M forward exact** 冲主胜负 | F1/A1 副赛道分表；novelty：副路可后置；loader 注释已写 GGUF 暂不进本 crate |
| **GPU / CUDA 进主表** | F4、A5、A8 |
| **logits / token tol 当 Phase1 门闩** | F2 |
| **加载期整模转 f32（或转 FP16）当主赛道默认** | 布局钉 BF16 存储；M_peak / 零拷贝 |
| **跨档混报胜负**（BF16 表 vs Q4 表） | A1/A2 |
| **用 S3+/S6/batch N/S 单独宣称打败 llama.cpp** | §4.3 |

Phase1 主目标仍是：**SafeTensors BF16 + CPU + S1 greedy exact vs llama.cpp**；Q4/GPU 不是本阶段计算策略议题。

---

## 5. 给实现的一句话收束

**【推测/建议】** 主路 = **mmap 上的 BF16 视图 + 算子内 f32 累加**；按 embed → RMSNorm → RoPE → GQA → SwiGLU → tied lm_head 补齐；指纹钉 `acc_path`；Q4/GPU/整表 f32 物化全部挡在 Phase1 门外。

---

*起草：深析工作代理 · 只分析 · 不合入 git（交 OSdeveloper）*
