# 刀⑪负结果收口 + 刀⑫口径（2026-11-04）

> **深析判决 · 同刀⑩**：刀⑪ host 回退，**不合**多投影融合进默认；账本记负结果；下一刀基线=刀⑨（禁用 11.58 / 11.81）。随后开刀⑫（跳出融合同族，也跳出 GEMV 形态同族）。不把刀⑪写成达标或改善。

## ① 刀⑪收口（硬门未过；端到端回退；融合不进默认）

- Host W=3/R=5，指纹 `s1-myos-bf16-t8-k11`，代码 `phase1/knife11-fused-decode@491260a`（PR #50，**已关，不合入**）。
- CORRECT=PASS、DET=PASS；L_TTFT 2180.442 ≤ 2193.6 过。
- **T_decode 11.5767 ≪ 16.37 / 16.42**；场景加权仍 **−80**。硬门未过。
- 相对刀⑨：T_decode **15.4880 → 11.5767（约 −25.3%）**。略差于刀⑩ 11.8074。同 x 融合的微基准不能当成端到端 tok/s；S1 墙钟是回退。

### 合入策略（同刀⑩）

- 刀⑨是「略快但不够」负结果，代码可合入且默认关预取作基线。
- 刀⑩、刀⑪都是**回退**：矩形切行与多投影融合 **都不进默认**。#45 不合；**#50 已关，不合**。main 保持刀⑨（`1460ef22` / PR #43）。刀⑩账本是 #47，只记负结果。
- 下一刀基线 = 刀⑨（T_decode 15.4880），**禁止用 11.58 / 11.81** 当对照或宣称改善。
- 账本：`arena/s1-myos-bf16-t8-k11-cost-card.md`、`arena/phase1-scoreboard.md`（记 FAIL+回退）。

| 硬门 | 门槛 | 实测 | 判 |
|------|------|------|----|
| CORRECT / DET | PASS @ threads=8 | PASS / PASS | 过 |
| T_decode 平带 | ≥16.42 或（≥16.37 且加权严格优于 −80） | 11.5767；加权 −80 | **未过（回退）** |
| L_TTFT | ≤2193.6 ms | 2180.442 | 过 |
| 相对刀⑨ | T_decode ≥ 15.4880 | 11.5767（约 −25.3%） | **回退** |

分项 vs llama `s1-llamacpp-bf16-b11960`（±5%）：

| 指标 | MyOS | llama | ratio | 判定 |
|------|------|-------|-------|------|
| T_decode | 11.5767 | 17.2348 | 0.6717 | L |
| L_TTFT | 2180.442 | 608.110 | 3.5856 | L |
| L_TPOT | 86.380 | 58.022 | 1.4887 | L |
| M_peak | 2963.496 | 3050.953 | 0.9713 | T（合成按协议） |
| 场景 | — | — | 加权 −80 | L |

## ② 刀⑫ · 跳出融合同族，也跳出 GEMV 形态同族（一口 Attention / GQA / KV 布局）

**目标**：在刀⑨基线上，一口改 **Attention / GQA / KV 布局**，把 T_decode 推进协议平带，从而场景加权 **严格优于 −80**。这一口离开多投影融合同族，也离开 GEMV 行块 / 寄存器块同族。

### 改动面

- Attention、GQA、KV 布局，一口
- 仍 BF16+fp32 acc、先乘后加、`threads=8`、`MYOS_GEMV_PACK` 默认关、预取默认关
- prefill 保持刀⑧ chunk；不回退串行 `forward_token`×n_prompt

### 硬验收

- S1 CORRECT + DET @ `threads=8`
- T_decode ≥ **16.42** tok/s **或** 至少 T_decode ≥ **16.37** 且场景加权 **严格优于 −80**
- L_TTFT ≤ **2193.6** ms
- **相对刀⑨** T_decode ≥ **15.4880**
- 指纹另开：`runs/s1-myos-bf16-t8-k12/`；更新 cost-card / scoreboard
- **不做** S2 / Q4。硬门过则要求加权严格优于 −80

### 不做

- 再切 GEMV 行宽 / 寄存器块宽
- 再做 fused multi-GEMV（Q/K/V 或 gate/up 同 x 一次算完）
- pack / 预取开关
- 单算子微刀；只调 threads；打乱乘加序
- 用 11.5767 / 11.8074 当改善基线
- 开 S2 / Q4

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
  --fingerprint-dir runs/s1-myos-bf16-t8-k12 \
  --cost-card arena/s1-myos-bf16-t8-k12-cost-card.md
```
