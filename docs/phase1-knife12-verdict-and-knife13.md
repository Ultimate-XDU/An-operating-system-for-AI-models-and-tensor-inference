# 刀⑫负结果收口 + 刀⑬口径（2026-11-04）

> **判决（深析）**：关 PR #53，**不合入**；按刀⑩/⑪同族负结果落账本；基线仍钉已合入刀⑨；另开刀⑬（禁 Attention/KV 同族微刀）。不把刀⑫写成达标、场景胜或相对⑨不回退。

## ① 刀⑫收口（硬门 FAIL；关 #53）

- Host W=3/R=5，指纹 `s1-myos-bf16-t8-k12`，测得 challenger `6a0e5c8`（卡/指纹一致；OSdeveloper 报 tip `9fcccaff` 若为同 PR 后续提交，以本卡中位为准判收口）。
- 改动面：KV 就地读（decode 不再每步 `pack_prefix_with_token`；prefill `attend_chunk` 同 stride 直读）。一口粗改，默认无新开关；未行切、未 fused multi-GEMV、pack/预取仍默认关。
- CORRECT=PASS、DET=PASS；L_TTFT **1560.521** ≤ **2193.6**（刀⑨×1.10）过，且相对刀⑨ TTFT 约 **−21.7%**（prefill 侧受益）。
- **T_decode 13.5677 < 15.4880（刀⑨）约 −12.4%**，亦远低于平带 16.37/16.42；L_TPOT 73.704；场景加权仍 **−80**。相对⑨不回退硬门 **FAIL** → 整刀硬门 **FAIL**。

### 对照结论（供以后改路径对照；非 exact bug）

- CORRECT/DET 仍 PASS：这是**墙钟回归负结果**，不是 exact 破损。
- 同族链条：刀⑩切行 ≈11.81（相对⑨约 −24%）、刀⑪融合 ≈11.58（约 −25%）、刀⑫ KV 直读 13.57（约 −12%）。⑫ 比⑩/⑪ 轻，但仍明显慢于⑨；**TTFT 改善不能兑换 decode 回退**（硬门显式要求相对⑨ `T_decode ≥ 15.4880`）。
- **合入策略**：关 #53，**不合核**。负结果账本另开（卡 `arena/s1-myos-bf16-t8-k12-cost-card.md`、本文件、scoreboard 脚注）；**不要**把⑫代码合进 main，基线继续用刀⑨中位。

| 硬门 | 门槛 | 实测 | 判 |
|------|------|------|----|
| CORRECT / DET | PASS @ threads=8 | PASS / PASS | 过 |
| T_decode 平带 | ≥16.42 或（≥16.37 且加权优于 −80） | 13.5677；加权 −80 | **未过** |
| L_TTFT | ≤2193.6 ms | 1560.521 | 过 |
| 相对刀⑨ | T_decode ≥ **15.4880** | 13.5677（约 −12.4%） | **未过** |

分项 vs llama `s1-llamacpp-bf16-b11960`（±5%）：

| 指标 | MyOS | llama | ratio | 判定 |
|------|------|-------|-------|------|
| T_decode | 13.5677 | 17.2348 | 0.7872 | L |
| L_TTFT | 1560.521 | 608.110 | 2.5662 | L |
| L_TPOT | 73.704 | 58.022 | 1.2703 | L |
| M_peak | 2962.938 | 3050.953 | 0.9712 | T（合成按协议） |
| 场景 | — | — | 加权 −80 | L |

相对刀⑨（合入基线）：

| 指标 | 刀⑨ | 刀⑫ | Δ |
|------|------|------|---|
| T_decode | 15.4880 | 13.5677 | −12.4% |
| L_TTFT | 1994.139 | 1560.521 | −21.7% |
| L_TPOT | 64.566 | 73.704 | +14.2% |
| M_peak | 2963.445 | 2962.938 | ≈平 |
| 场景加权 | −80 | −80 | 未翻 |

## ② 刀⑬ · 一口粗块：tied lm_head / vocab 投影（禁同族微刀）

**目标**：在刀⑨基线上，一口砍 decode 最大非对称热路径——tied `embed_tokens` 作 vocab 投影（`[vocab=151936, hidden=1536]`，每步 decode 一次），把 `T_decode` 推进平带并使场景加权 **优于 −80**。

**依据（可核对）**：⑤–⑦ 已把通用 `gemv_bf16`（AVX2 多行 + threads=8）铺到主路径；⑥ pack / ⑨ 预取开关 / ⑩ 行切 / ⑪ fused multi-GEMV / ⑫ Attention·KV 布局均未兑现墙钟或回退。decode 仍每 token 打一次 vocab 宽 GEMV，与层内 `[out, 1536]` 投影不对称；此刀换**算子族**，不再在 KV/行块/多投影融合上打转。

### 改动面

- 仅默认路径一口：tied lm_head（`EMBED_TOKENS` 视图上的 vocab GEMV）的 decode 热路径粗改——允许同数值语义下的分块/寄存器阻塞/与现有 `gemv_bf16` 同序的专用核，**禁止**改乘加序、禁止 FMA 替换先乘后加、禁止打乱 greedy 平局规则。
- 基线对照：main 刀⑨中位；**禁用**⑩/⑪/⑫ 回退中位作对照。
- prefill 保持刀⑧ chunk；层内 QKV/O/MLP/GQA/RoPE/RMSNorm **本刀不改**。

### 硬验收（同前，且相对⑨不回退）

- S1 CORRECT + DET @ `threads=8`
- T_decode ≥ **16.42** tok/s（期望 L_TPOT ≤ 60.92）**或** 至少 T_decode ≥ **16.37** 且场景加权 **优于 −80**
- L_TTFT ≤ **2193.6** ms（刀⑨×1.10）
- 相对刀⑨：T_decode ≥ **15.4880**（不回退）
- 指纹另开：`runs/s1-myos-bf16-t8-k13/`；卡 `arena/s1-myos-bf16-t8-k13-cost-card.md`
- **不要求**本刀场景胜 / 开 S2 / Q4；硬门过则要求加权优于 −80

### 不做（禁同族微刀）

- Attention / GQA / KV stride、`pack_prefix*`、就地读变体、softmax 微形态
- 行切 / 寄存器块（⑩ 族）、fused multi-GEMV / 多投影融合（⑪ 族）
- `MYOS_GEMV_PACK` / `MYOS_GEMV_PREFETCH` 开关级；只调 threads；打乱乘加序；开 S2/Q4
- 单算子微刀：RoPE / RMSNorm / SiLU  alone

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
  --fingerprint-dir runs/s1-myos-bf16-t8-k13 \
  --cost-card arena/s1-myos-bf16-t8-k13-cost-card.md
```

### 局限（写进账本）

- 若 profile 显示 vocab GEMV 占比不足以解释距平带的 ≈6%（相对⑨），本刀仍可能负；那时下一刀再换族，**不**退回⑫族微改。
- 微基准胜 ≠ 墙钟；收口只认本指纹 W=3/R=5 中位。
