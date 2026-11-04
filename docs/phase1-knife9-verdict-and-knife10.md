# 刀⑨负结果收口 + 刀⑩口径（2026-11-04）

> **月星拍板（2026-11-04）**：刀⑨按**负结果收口**合入 PR #43；随后开刀⑩（更大块 decode 粗改）。不把刀⑨写成达标或场景胜。

## ① 刀⑨收口（硬门未过；负结果合入）

- Host W=3/R=5，指纹 `s1-myos-bf16-t8-k9`，代码 `595943d` + 账本 `daa3a44`（≈08:48–08:50 UTC+8）。
- CORRECT=PASS、DET=PASS；L_TTFT 1994.139 ms ≲ 2413（刀⑧×1.10）过。
- **T_decode 15.4880 < 16.37 / 16.42**，L_TPOT 64.566 > 60.92；场景加权仍 **−80**。硬门未过。
- 相对刀⑧：T_decode 14.9912→15.4880（约 **+3.3%**），不够翻 ±5% 平带。

### 对照结论（供以后改路径对照；非 bug）

- 本刀默认关掉 decode GEMV **软件预取**（`MYOS_GEMV_PREFETCH=1` 可再开）；pack 仍默认关。
- 云端微基准上，关预取对大于 L3 的 GEMV 可约 +15%–25%；**端到端 S1 decode 仅约 +3%**，未进协议 ±5% 平带。
- **这是负结果收口，不是回归 / exact 破损 / 实现 bug**：CORRECT/DET 仍 PASS；预取默认关作为可复现基线保留，供后续粗改对照。
- 账本：`arena/s1-myos-bf16-t8-k9-cost-card.md`、`arena/phase1-scoreboard.md`。

| 硬门 | 门槛 | 实测 | 判 |
|------|------|------|----|
| CORRECT / DET | PASS @ threads=8 | PASS / PASS | 过 |
| T_decode 平带 | ≥16.42 或（≥16.37 且加权优于 −80） | 15.4880；加权 −80 | **未过** |
| L_TTFT | ≲2413 ms | 1994.139 | 过 |

分项 vs llama `s1-llamacpp-bf16-b11960`（±5%）：

| 指标 | MyOS | llama | ratio | 判定 |
|------|------|-------|-------|------|
| T_decode | 15.4880 | 17.2348 | 0.8986 | L |
| L_TTFT | 1994.139 | 608.110 | 3.2792 | L |
| L_TPOT | 64.566 | 58.022 | 1.1128 | L |
| M_peak | 2963.445 | 3050.953 | 0.9713 | T（合成按协议） |
| 场景 | — | — | 加权 −80 | L |

## ② 刀⑩ · 更大块 decode 粗改（一口）

**目标**：在刀⑨「预取默认关」基线上，一口做更大块 decode 粗改，把 T_decode 推进协议平带，从而场景加权 **优于 −80**。禁止再玩开关级微刀（预取/pack 开关）。

### 改动面

- 行切 / 寄存器阻塞 / 多 token 同权重复用等粗改（仍 BF16+fp32 acc、先乘后加、`threads=8`、`MYOS_GEMV_PACK` 默认关、预取默认关）
- prefill 保持刀⑧ chunk；不回退串行 `forward_token`×n_prompt
- 非同族二次 pack；非 RoPE/RMSNorm 等单算子微刀；不靠再拨 `MYOS_GEMV_PREFETCH`

### 硬验收

- S1 CORRECT + DET @ `threads=8`
- T_decode ≥ **16.42** tok/s（期望 L_TPOT ≤ 60.92）**或** 至少 T_decode ≥ **16.37** 且场景加权 **优于 −80**
- L_TTFT 不差于刀⑨ 中位的 **1.10×**（≲ **2194** ms）
- 指纹另开：`runs/s1-myos-bf16-t8-k10/`；更新 cost-card / scoreboard
- **不要求**本刀场景胜 / 开 S2 / Q4；**要求**加权优于 −80（若硬门过）

### 不做

- 再玩预取/pack 开关级微刀；单算子微刀；只调 threads；打乱乘加序；开 S2/Q4

### 交付

- 一块 PR；host 补测同形：

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
  --fingerprint-dir runs/s1-myos-bf16-t8-k10 \
  --cost-card arena/s1-myos-bf16-t8-k10-cost-card.md
```
