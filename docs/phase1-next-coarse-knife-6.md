# Phase1 刀⑥口径 · 主路径带宽粗块

| 项 | 内容 |
|----|------|
| 文档 | `docs/phase1-next-coarse-knife-6.md` |
| 前提 | 刀⑤已收口：⑤a `bb91cdc` S1 WALL≈85.7s CORRECT；⑤b main@`840e734b` S1 WALL≈94.2s CORRECT；closeout #35 @ `7529005c`；联调 R17 PASS |
| 效力 | 与协议冲突时以 `PHASE1-ARENA-PROTOCOL-v0.1.md` 为准；与刀⑤收口冲突时以本刀「与⑤边界」为准 |

## 目标

在 **不改数值语义** 的前提下，压主路径对权重流的带宽/缓存浪费，让 S1（128+128，`threads=1`）墙钟相对刀⑤a 收口基线再跨一档，为后续正式 cost-card 对打留余地。

**不**把本刀做成「再加宽 YMM 行数」；⑤b≈⑤a 已说明行并行微形态收益已尽。

## 主路径改动点（粗块，只选这一条主推）

**对 `gemv_bf16` / `gemv_bf16_kernel` 热路径做「BF16 面板打包（tile scratch）± 软件预取」：**

1. **面板打包（主）**：按现有多行 tile（与⑤核同一行组宽度即可）把即将计算的权重面板从 mmap 行主序 **临时** 收成对列遍历更友好的 BF16 字节布局，写入 **有界 scratch**（L2/L3 量级面板，禁止整模第二份常驻副本）。累加仍是每行 `j=0..k-1`、先乘后加、非 FMA，与 scalar `dot_row` 同序同比特。
2. **软件预取（辅，可同 PR）**：对下一列组 / 下一面板发 `_mm_prefetch`（或等价），不改运算序。
3. 接线：仍经现有 `gemv_bf16` API；`model_core` 前向控制流、键名、RoPE/QKV bias/SwiGLU **不动**。
4. 权重源：SafeTensors **仍 mmap BF16 字节视图**；打包只发生在算子内 scratch，**禁止**加载期整表 `Vec<f32>`，**禁止**把整模 BF16 复制成第二份常驻镜像（会直接打爆 M_peak 叙事）。

## 验收门（exact / CORRECT **仍硬**）

| # | 门 | 硬？ |
|---|----|------|
| 1 | S1 CORRECT：同 E2 配对 vs `runs/s1-llamacpp-bf16-b11960/token_ids.txt`，128/128 PASS | **硬** |
| 2 | 指纹：`quant_or_dtype=bf16`，`acc_path=bf16_w_fp32_acc`；deterministic 主路径 | **硬** |
| 3 | 数值序：每行左→右 f32 链、乘加非 FMA；与⑤ reference scalar 比特一致（可用既有/扩展单测钉死） | **硬** |
| 4 | 墙钟：同机 S1 128+128、`threads=1`，相对 **⑤a 基线 WALL≈85.7s** **≥2×**（墙钟 ≲ **42.9s**）。以⑤a 为对照（⑤b 略慢，过此门同时优于⑤b） | **硬（本刀吞吐门）** |
| 5 | no_std：打包/预取核在 `tensor_core`；mmap/CLI/指纹仍在 runtime | **硬** |
| 6 | 回归：cycle smoke + 微型 ST 自洽仍绿 | **硬** |

**不要求**：本刀已胜 llama ±5%；正式 W=3/R=5 cost-card（另派）；打开 S2。

若面板打包后 CORRECT 与 ≥2× **不可兼得**，优先保 CORRECT，PR 内改为「仅预取」并在笔记写明未过 2×——**不得**用松数值换速度。

## 不做清单

| 不做 | 原因 |
|------|------|
| 再加宽 SIMD 行数 / 新开 64 行核 | 属刀⑤微形态，⑤b 已示收益薄 |
| 先开 S2/S3 场景 | 月星/既有口径：不先 S2 |
| 副赛道 Q4 对打 | 仍不开 |
| rayon / 多线程破 deterministic exact | exact 硬门 |
| 加载期整表 f32 或整模常驻第二份权重 | 破 BF16 视图门闩与 M_peak |
| 换 NeoX / 去 QKV bias / 改 SwiGLU / FMA 换乘加序 | 破 exact |
| 细算子超市（逐个 RMSNorm/RoPE 微优） | 非本粗块 |
| 本刀宣称打败 llama.cpp | 只收 CORRECT + ≥2× |

## 与刀⑤的边界

| | 刀⑤（已收口） | 刀⑥（本刀） |
|--|----------------|-------------|
| 问题 | 算不足：naive 标量 → 同序多行/SIMD | 带宽/缓存：行主序 mmap 流 + tile 访存形态 |
| 手段 | AVX2 多行累加、同布局 | BF16 **面板 scratch 打包** ± 软件预取 |
| 布局 | 不改权重内存布局 | 允许算子内 **临时** 面板布局；源仍 mmap 行主序 |
| 收口信号 | ≥10× vs E2 31min + CORRECT | ≥2× vs ⑤a 85.7s + CORRECT |
| 禁止互吞 | ⑥ 不得再以「加行数」交差 | ⑤ 文档不再追加预取/打包 |

## CloudAgent

**适合开干。** Steer 须钉：CORRECT 硬；乘加序硬；禁止整模二份副本；PR 附 S1 CORRECT + 墙钟前后（对照 85.7s）与是否启用 pack/prefetch 的开关说明。
