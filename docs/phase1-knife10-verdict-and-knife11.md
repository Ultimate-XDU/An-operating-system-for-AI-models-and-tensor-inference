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

## ② 刀⑪ · 跳出 GEMV 行块同族（一口）

**目标**：在刀⑨基线上，一口做**更大块** decode 粗改——优先**层内/跨步权重复用**或**多投影融合**（刀⑩未落地的粗条），把 T_decode 推进协议平带，从而场景加权 **优于 −80**。禁止再切行宽 / 预取 / pack。

### 改动面

- 层内或跨 decode 步的权重复用；或多投影（如 Q/K/V 或 gate/up）同权重视图融合 / 一次加载多算
- 仍 BF16+fp32 acc、先乘后加、`threads=8`、`MYOS_GEMV_PACK` 默认关、预取默认关
- prefill 保持刀⑧ chunk；不回退串行 `forward_token`×n_prompt

### 硬验收

- S1 CORRECT + DET @ `threads=8`
- T_decode ≥ **16.42** tok/s（期望 L_TPOT ≤ 60.92）**或** 至少 T_decode ≥ **16.37** 且场景加权 **优于 −80**
- L_TTFT ≤ **2193.6** ms（刀⑨×1.10）
- **相对刀⑨不回退**（T_decode ≥ 15.4880）
- 指纹另开：`runs/s1-myos-bf16-t8-k11/`；更新 cost-card / scoreboard
- **不要求**本刀场景胜 / 开 S2 / Q4；硬门过则要求加权优于 −80

### 不做

- 再切 GEMV 行宽 / 寄存器块宽；预取/pack 开关微刀；单算子微刀；只调 threads；打乱乘加序；开 S2/Q4

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
  --fingerprint-dir runs/s1-myos-bf16-t8-k11 \
  --cost-card arena/s1-myos-bf16-t8-k11-cost-card.md
```

刀⑪ host 负结果与刀⑫口径见 `docs/phase1-knife11-verdict-and-knife12.md`（#50 不合入；基线仍刀⑨ T_decode 15.4880）。
