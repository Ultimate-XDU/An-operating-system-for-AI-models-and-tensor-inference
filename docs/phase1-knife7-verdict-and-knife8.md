# 刀⑦判决 + 刀⑧口径（深析 2026-11-04）

## ① 刀⑦收口

- **收口，建议合 #39。**
- 硬门是可比闭环：`threads=8` + cost-card + scoreboard + CORRECT/DET，**不是本刀必胜**。
- 本机 W=3/R=5（2026-11-03 15:02 UTC+8）：CORRECT=PASS、DET=PASS、comparable=true、场景 **L**（加权和 −80）。
  - T_decode 14.3503 vs llama 17.2348（0.83×）
  - L_TTFT 9092.903 ms vs 608.110（≈15×，场景负主因）
  - L_TPOT 69.685 vs 58.022
  - M_peak 2963.383 vs 3050.953（T）
- GEMV t1→t8 ≈3.611× 仅软记录，不作 S1 ±5% 结论。

## ② 刀⑧ · Prefill/TTFT 主路径（一口）

**目标**：吃整段 prompt prefill，禁止拆小算子 / 二次 pack。

**依据**：L_TTFT≈15× 是场景负主因；T_decode 已 0.83×。现状 `prefill`=串行 `forward_token`×n_prompt，与 llama prefill 不在同一结构档。

### 改动面

- prompt-parallel / 分块 prefill（投影 + 注意力写 KV），非 `n_prompt` 次 decode 同构
- decode 可继续刀⑦的 `threads=8`
- 仍 BF16 视图 + fp32 acc；exact 与金标准一致

### 硬验收

- S1 CORRECT + DET @ `threads=8`
- L_TTFT ≤ **3000 ms** **或** 相对刀⑦ ≥**3×** 改善（≲3031 ms）
- T_decode ≥ 刀⑦ 的 **0.90×**（≥12.9 tok/s）
- 场景加权和优于 −80
- 更新 cost-card / scoreboard
- **不要求**本刀场景胜 / 开 S2 / Q4

### 不做

- pack；RoPE/RMSNorm 微刀；只刷 threads 充⑧；打乱乘加序

### 交付

- 一块 PR；口径原文即本文
- host 补测命令与刀⑦同形，指纹目录另开（如 `runs/s1-myos-bf16-t8-k8/`）

## 落地

基线 main 刀⑦ `a92677f1`。本环境没有 1.5B 权重，CORRECT / DET / 中位记 **UNMEASURED**。单测已经把 chunk prefill 和 serial `forward_token` 的 logits、KV 按 f32 bits 对齐。

`prefill` 一次吃完整段 prompt。每一层对 chunk 做 Q/K/V/O 与 SwiGLU 的 `gemm_bf16`：权重保持 BF16 视图，不 pack；每个 token 的每一行仍是 `j=0..k-1` 先乘后加，非 FMA。K/V 写入该层后，每个位置调用现有 `gqa_attn_f32`（`GqaMask::Causal`）。tied head 只跑最后一个 prompt token。`greedy_decode` 仍走 `forward_token`，threads=8 行维并行。`MYOS_GEMV_PACK` 默认关。协议 batch 仍是 1。

host 补测与刀⑦同形，指纹目录另开。不要设 `MYOS_GEMV_PACK`。

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
  --fingerprint-dir runs/s1-myos-bf16-t8-k8 \
  --cost-card arena/s1-myos-bf16-t8-k8-cost-card.md
```

产物：`runs/s1-myos-bf16-t8-k8/metrics.json`（中位）、`det_evidence.json`、`token_ids.txt`、`tokens.txt`、`cost-card.md`。CORRECT 或 DET 失败时进程退出码 1。

硬门在 host 填中位之后再判，本环境不把 UNMEASURED 写成已经 ≤3000ms 或已经场景胜。门槛是：S1 CORRECT+DET @ threads=8；L_TTFT ≤3000ms 或相对刀⑦ 9092.903ms ≥3×（≲3031ms）；T_decode ≥12.9 tok/s；场景加权和优于 −80。不要求本刀场景胜，不开 S2，不开 Q4。
