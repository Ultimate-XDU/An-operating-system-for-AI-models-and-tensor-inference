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

## ② 刀⑫ · Attention 按 cache stride 直读已落地（端到端 UNMEASURED）

一口粗改，默认路径没有新开关。对照基线是刀⑨ host 中位 **T_decode 15.4880**（指纹 `s1-myos-bf16-t8-k9`），禁止用刀⑪的 11.5767 或刀⑩的 11.8074 当对照。这一口离开多投影融合同族，也离开 GEMV 行块 / 寄存器块同族。

### 改动面

- `KvCache` 仍是 `[num_kv_heads, max_seq, head_dim]`。decode 不再每步 `pack_prefix_with_token`。已提交的前缀按 `kv_stride = max_seq` 直接读；当前 token 的 K/V 放在旁边，注意力接受之后才 `store`。不整段 pack。
- prefill 仍是刀⑧的整段 chunk。K/V 已经 `store_at` 进 cache，`attend_chunk` 用同一 stride 读，不再 `pack_prefix`。因果掩码仍是前缀。不回退串行 `forward_token`×n_prompt。
- `gqa_attn_f32` 的点积、softmax、每条输出 lane 的累加顺序不变：先乘后加，不 `mul_add`，softmax 仍是 max、`expf(score - max)`、从左到右求和、`e / sum`。同一 KV head 的 query（1.5B 为 6）一起走这些链；V 的多条 lane 一起走，每条 lane 仍按 token 从左到右加。
- 仍 BF16+fp32 acc、`threads=8`、`MYOS_GEMV_PACK` 默认关、预取默认关。不改 GEMV 行宽 / 寄存器块，不做 fused multi-GEMV，不改乘加序，不调 threads，不开 S2 / Q4。

### 硬门（host 补测；本环境无 1.5B 权重）

| 硬门 | 门槛 |
|------|------|
| CORRECT / DET | PASS @ threads=8 |
| T_decode | ≥ **16.42**，或（≥ **16.37** 且场景加权严格优于 −80） |
| L_TTFT | ≤ **2193.6** |
| 相对刀⑨ | T_decode ≥ **15.4880** |

指纹目录：`runs/s1-myos-bf16-t8-k12/`。账本草稿：`arena/s1-myos-bf16-t8-k12-cost-card.md`。端到端 S1 在写这份草稿的环境里是 **UNMEASURED**。上表没有填实测中位。不把空位读成平带或场景胜。**不做** S2 / Q4。硬门过则要求加权严格优于 −80。

### 不做

- 再切 GEMV 行宽 / 寄存器块宽
- 再做 fused multi-GEMV（Q/K/V 或 gate/up 同 x 一次算完）
- pack / 预取开关
- 单算子微刀；只调 threads；打乱乘加序
- 用 11.5767 / 11.8074 当改善基线
- 开 S2 / Q4

### host 补测

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

硬门在 host 填中位之后再判。不在本刀开 S2 / Q4。
