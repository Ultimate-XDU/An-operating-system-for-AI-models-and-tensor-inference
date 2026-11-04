# 刀⑩负结果收口 + 刀⑪口径（2026-11-04）

> **深析判决 · OSdeveloper 采纳**：刀⑩ host 回退，**不合**矩形切行进默认；账本记负结果；下一刀基线=刀⑨（禁用 11.81）。随后开刀⑪（跳出 GEMV 行块同族）。不把刀⑩写成达标或改善。

## ① 刀⑩收口（硬门未过；端到端回退；切行不进默认）

- Host W=3/R=5，指纹 `s1-myos-bf16-t8-k10`，代码 `0086dd05`（PR #45，**不合入**）。
- CORRECT=PASS、DET=PASS；L_TTFT 2042.858 ≤ 2193.6 过。
- **T_decode 11.8074 ≪ 16.37 / 16.42**；场景加权仍 **−80**。硬门未过。
- 相对刀⑨：T_decode **15.4880 → 11.8074（约 −23.8%）**——不是⑨那种「+3% 不够」，是端到端回退。
- 微基准「8 行寄存器块更快」被 S1 墙钟证伪。

### 合入策略（异于刀⑨）

- 刀⑨是「略快但不够」负结果，代码可合入且默认关预取作基线。
- 刀⑩是**回退**：矩形切行 **不进默认**；**#45 不合**；main 保持刀⑨。
- 下一刀基线 = 刀⑨（`1460ef22` / T_decode 15.4880），**禁止用 11.81** 当对照或宣称改善。
- 账本：`arena/s1-myos-bf16-t8-k10-cost-card.md`、`arena/phase1-scoreboard.md`（记 FAIL+回退）。

| 硬门 | 门槛 | 实测 | 判 |
|------|------|------|----|
| CORRECT / DET | PASS @ threads=8 | PASS / PASS | 过 |
| T_decode 平带 | ≥16.42 或（≥16.37 且加权优于 −80） | 11.8074；加权 −80 | **未过（回退）** |
| L_TTFT | ≤2193.6 ms | 2042.858 | 过 |
| 相对刀⑨ | 不回退 | −23.8% | **回退** |

分项 vs llama `s1-llamacpp-bf16-b11960`（±5%）：

| 指标 | MyOS | llama | ratio | 判定 |
|------|------|-------|-------|------|
| T_decode | 11.8074 | 17.2348 | 0.6851 | L |
| L_TTFT | 2042.858 | 608.110 | 3.3594 | L |
| L_TPOT | 84.693 | 58.022 | 1.4597 | L |
| M_peak | 2963.469 | 3050.953 | 0.9713 | T（合成按协议） |
| 场景 | — | — | 加权 −80 | L |

## ② 刀⑪ · 多投影融合已落地（端到端 UNMEASURED）

一口，默认开，没有加速开关（没有 `MYOS_FUSE` 一类回退）。pack 仍默认关，软件预取仍默认关（`MYOS_GEMV_PREFETCH=1` 才开）。行宽和寄存器块宽不动。乘加序不动：每行仍是 `j = 0..k-1` 先乘后加，非 FMA，无跨行归约。对照基线是刀⑨ host 中位 **T_decode 15.4880**（指纹 `s1-myos-bf16-t8-k9`），禁止用刀⑩的 11.8074 当对照。

### 改动面

1. **多投影融合。** Decode 的 Q、K、V 共用同一次激活，一次 worker 波算完再加 bias。gate 与 up 同样：一次波，然后 SiLU，down 仍是下一次 GEMV（它读的是 `silu(gate)*up`）。每个 worker 先走自己那一段第一张矩阵的行，再走下一张，中间不 join。数值核仍是刀⑨那份按行的 BF16→f32 点积。
2. **`greedy_decode` 权重视图复用。** 在完整权重上把每一层的 BF16 视图借一次，后面的 step 复用这些切片，不再逐步重查名字。字节仍按步从映射里流过，不复制成 f32，也不 pack。缺张量时退回逐层查找，以便较早的层仍能按原错误路径写入 KV。

Prefill 保持刀⑧的整段 chunk（`gemm_bf16`），不走这条 decode 波。仍 BF16+fp32 acc、`threads=8`、`MYOS_GEMV_PACK` 默认关。

### 硬门（host 补测；本环境无 1.5B 权重）

| 硬门 | 门槛 |
|------|------|
| CORRECT / DET | PASS @ threads=8 |
| T_decode | ≥16.42，或（≥16.37 且场景加权严格优于 −80） |
| L_TTFT | ≤2193.6 |
| 相对刀⑨不回退 | T_decode ≥ **15.4880** |

指纹目录：`runs/s1-myos-bf16-t8-k11/`。账本草稿：`arena/s1-myos-bf16-t8-k11-cost-card.md`。端到端 S1 在写这份草稿的环境里是 **UNMEASURED**。上表没有填实测中位。不要求本刀场景胜或开 S2 / Q4；硬门过则要求加权优于 −80。

### 微基准（不是 S1）

`cargo test -p tensor_core --release release_same_x`，本机 release，threads=8，权重反复打在同一块缓冲上（热缓存，不是 mmap 冷流）。两次：

| 组 | 形状 | separate / fused |
|----|------|------------------|
| Q/K/V | `[1536,1536]` + `[256,1536]` + `[256,1536]` | 1.185，1.164 |
| gate/up | 两个 `[8960,1536]` | 1.132，1.137 |

比值大于 1 表示融合波更短。这组数不能乘到 15.4880 上当成端到端 tok/s。S1 的权重大约 2.8GiB，不会整段留在缓存里。

### 不做

- 再切 GEMV 行宽 / 寄存器块宽；预取/pack 开关微刀；单算子微刀；只调 threads；打乱乘加序；开 S2/Q4
- 用刀⑩ 11.8074 当改善基线

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
  --fingerprint-dir runs/s1-myos-bf16-t8-k11 \
  --cost-card arena/s1-myos-bf16-t8-k11-cost-card.md
```

硬门在 host 填中位之后再判。不在本刀开 S2 / Q4。
