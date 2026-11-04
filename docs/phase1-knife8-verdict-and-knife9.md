# 刀⑧判决 + 刀⑨口径（深析 2026-11-04）

## ① 刀⑧收口（条件）

- **条件收口，建议合 #41**（或同分支 `phase1/knife8-prefill-ttft` @ `16afd92`）。
- Prefill/TTFT **主目标硬门全过**；硬验收清单中 **「场景加权和优于 −80」未过**（仍 −80 / 场景 L）。账本须钉「TTFT 达标、场景分未改善」，**不对外宣称场景改善或场景胜**。
- 本机 W=3/R=5（指纹 `s1-myos-bf16-t8-k8`，commit `16afd92`）：CORRECT=PASS、DET=PASS、comparable=true。

| 硬门 | 门槛 | 实测 | 判 |
|------|------|------|----|
| CORRECT / DET | PASS @ threads=8 | PASS / PASS | 过 |
| L_TTFT | ≤3000 ms **或** vs⑦ ≥3×（≲3031） | 2193.653 ms（≈4.14× vs 9092.903） | 过 |
| T_decode | ≥ 刀⑦ 0.90×（≥12.9） | 14.9912 | 过 |
| 场景加权 | **优于 −80** | **仍 −80**（L/L/L/平） | **未过** |

分项 vs llama `s1-llamacpp-bf16-b11960`（±5% 协议）：

| 指标 | MyOS | llama | ratio | 判定 |
|------|------|-------|-------|------|
| T_decode | 14.9912 | 17.2348 | 0.8698 | L |
| L_TTFT | 2193.653 | 608.110 | 3.6073 | L |
| L_TPOT | 66.706 | 58.022 | 1.1497 | L |
| M_peak | 2963.520 | 3050.953 | 0.9713 | **平**（卡上「T」为方向标签；合成按协议 ±5%→0） |
| 场景 | — | — | 加权 −0.35−0.25−0.20+0 = **−0.80（−80）** | **L** |

相对刀⑦：TTFT 9092.9→2193.7（约 4.1×）；T_decode 14.35→14.99；四项带内判定未翻，故加权未优于 −80。`gemv_pack=false`；非 pack/微刀路径。

## ② 刀⑨ · Decode 吞吐翻 ±5% 平带（一口）

**目标**：一口把 decode 推到协议平带，从而 **场景加权优于 −80**（并尽量连带 L_TPOT 平）。禁止 pack / 单算子微刀 / 只刷 threads 充⑨。

**依据**：⑧ 已把 TTFT 从 ~15× 收到 ~3.6×，场景分仍卡在三负；离「优于 −80」最近的是 decode 侧。T_decode 与 L_TPOT 互为倒数：

- T_decode **平**：≥ 17.2348 × 0.95 ≈ **16.37** tok/s
- L_TPOT **平**：≤ 58.022 × 1.05 ≈ **60.92** ms → 约需 T_decode ≥ **16.42**
- 现状 14.99 → 约 **+9.5%** 墙钟即可双翻（或至少翻 T_decode）；加权可由 −80 升到约 −20（M 平）或 0（若 M 亦胜）

TTFT 再抠（2193→≲638 才翻平）量级更大，不作为本刀主路径。

### 改动面

- 整段 decode GEMV/多行/缓存友好粗改（沿刀⑤/⑦ 行维并行骨架放大），非同族二次 pack、非 RoPE/RMSNorm 微刀
- 仍 BF16 视图 + fp32 acc；乘加序与 exact 金标准一致；`threads=8`；`MYOS_GEMV_PACK` 默认关
- prefill 保持刀⑧ chunk 路径，本刀不回退串行 `forward_token`×n_prompt

### 硬验收

- S1 CORRECT + DET @ `threads=8`
- T_decode ≥ **16.42** tok/s（平带；同时期望 L_TPOT ≤ 60.92）**或** 至少 T_decode ≥ 16.37 且场景加权 **优于 −80**
- L_TTFT 不差于刀⑧ 中位的 **1.10×**（≲2413 ms；禁止 decode 优化反噬 prefill）
- 更新 cost-card / scoreboard（指纹另开，如 `runs/s1-myos-bf16-t8-k9/`）
- **不要求**本刀场景胜 / 开 S2 / Q4；**要求**加权优于 −80

### 不做

- pack；单算子微刀；只调 threads；打乱乘加序；开 S2/Q4

### 交付

- 一块 PR；口径原文即本文
- host 补测与刀⑧同形，指纹目录另开

```bash
cargo build -p myos_runtime --release
./target/release/myos \
  --arena-s1 \
  --weights /workspace/MyOS/weights/Qwen2.5-1.5B-Instruct/model.safetensors \
  --prompt-ids arena/prompts/s1_prompt.token_ids.txt \
  --ref-ids runs/s1-llamacpp-bf16-b11960/token_ids.txt \
  --steps 128 \
  --threads 8 \
  --warmup 3 \
  --repeats 5 \
  --fingerprint-dir runs/s1-myos-bf16-t8-k9 \
  --cost-card arena/s1-myos-bf16-t8-k9-cost-card.md
```

硬门在 host 填中位之后再判。
