# Cost Card · S1 · MyOS BF16 threads=8

### 设定
- fingerprint: `s1-myos-bf16-t8-k10`
- model: Qwen2.5-1.5B-Instruct
- track: SafeTensors BF16 (main track)
- device: cpu, n_threads: 8
- batch: 1
- quant_or_dtype: bf16
- acc_path: bf16_w_fp32_acc
- gemv_pack: false
- deterministic: ON (greedy argmax). seed: null (no RNG)
- n_ctx=256, n_prompt=128, n_gen=128
- protocol: W=3 discard, R=5 median
- challenger commit: null
- baseline: llama.cpp b11960 `s1-llamacpp-bf16-b11960` threads=8
- baseline source: `arena/e1-s1-bf16-cost-card.md` (T_decode 17.2348, L_TTFT 608.110 ms, L_TPOT 58.022 ms, M_peak 3050.953 MiB)
- cmdline: `./target/release/myos --arena-s1 --weights /workspace/MyOS/weights/Qwen2.5-1.5B-Instruct/model.safetensors --prompt-ids arena/prompts/s1_prompt.token_ids.txt --ref-ids runs/s1-llamacpp-bf16-b11960/token_ids.txt --steps 128 --threads 8 --warmup 3 --repeats 5 --fingerprint-dir runs/s1-myos-bf16-t8-k10 --cost-card arena/s1-myos-bf16-t8-k10-cost-card.md`
- measurement: **UNMEASURED** on this machine (no 1.5B weight file in the authoring environment). The harness below is what the host runs.

### 结果（中位数）
| metric | MyOS | llama.cpp | ratio (MyOS/base) | verdict |
|--------|------|-----------|-------------------|---------|
| T_decode (tok/s) | UNMEASURED | 17.2348 | — | UNMEASURED |
| L_TTFT / L_prefill (ms) | UNMEASURED | 608.110 | — | UNMEASURED |
| L_TPOT (ms) | UNMEASURED | 58.022 | — | UNMEASURED |
| M_peak (MiB RSS) | UNMEASURED | 3050.953 | — | UNMEASURED |
| t_load (s, not in T_decode) | UNMEASURED | — | — | appendix |
| T_e2e (tok/s) | UNMEASURED | — | — | appendix |

### 正确性
- greedy exact match: UNMEASURED
- det consistency: UNMEASURED

### 代价拆解
- 算力：decode GEMV 按矩形切行。高而 K 短（gate/up/tied head）仍是 32 行块；长 K 与方阵（down、Q/K/V/O）走 8 行寄存器块（一段 YMM，每次 8 列）。分给 8 个 worker。prefill 把整段 prompt 做成一次分块投影（同一权重行乘上 chunk 内每个 token）。每个 (token, 行) 仍是 `j=0..k-1` 先乘后加，非 FMA，无跨行归约，不 pack 权重。软件预取默认关（`MYOS_GEMV_PREFETCH=1` 才开）。刀⑩不拨预取/pack 开关。
- 内存：`M_peak` 是加载完成后、prefill 与 decode 期间的 VmRSS 峰值。权重保持 BF16 视图，无整表 f32。
- 工程：数值核 no_std；`std` 线程池只做确定行分割。刀⑥ panel pack 默认关（`gemv_pack` 见上）。
- 功能债：协议 batch 仍为 1。本卡不含 S2/S3，不含 Q4。

### 结论一句
- 胜负：不可比。本机没有 1.5B 权重，MyOS 中位是 UNMEASURED。llama 分母已写在表里，host 重跑后按 §4.2 填 W/T/L。
- 主要代价：行维并行对齐 llama `-t 8`；不把未测数字写成 ±5% 胜负。
- 是否值得为 Phase2 继续加码：S1 闭环先落地。Phase2 与否看 host 填完本卡之后的场景分，不在本刀宣布。

### 备注
刀⑩粗块：按矩形切 decode GEMV 的行宽。云端 release 同次微基准里，gate/up（`m≥4096` 且 `k≤2048`）上 32 行块更快，所以 gate、up、tied head 仍走刀⑨的 32 行块。down（长 K）约 1.10×、Q/O（方阵）约 1.18×，这两类改走 8 行寄存器块：一个 YMM 托 8 行，一次 8 列，列临时值用完即弃。乘加序仍是先乘后加、从左到右。`MYOS_GEMV_PACK` 与 `MYOS_GEMV_PREFETCH` 都保持默认关，本刀没有拨这两个开关。prefill 仍是刀⑧整段 chunk，没有退回 `forward_token`×n_prompt。

硬门（host 填中位后再判）：CORRECT+DET @ threads=8；T_decode ≥ 16.42（期望 L_TPOT ≤ 60.92）或（T_decode ≥ 16.37 且场景加权优于 −80；恰好 −80 不算过）；L_TTFT ≤ 刀⑨×1.10 = 2193.6 ms。硬门过则要求加权优于 −80，不要求场景胜。

### 计量边界（协议 §2.1）
1. `t_load` 单独计，不进入 L_prefill / T_decode。
2. `t_prefill`：从提交 prompt tokens 到产出第一个新 token 之前（含 KV / workspace 分配与整段 chunk prefill，不是 `n_prompt` 次 decode 同构 forward）。
3. `t_decode`：`greedy_decode` 跑完 `n_gen` 个新 token。`T_decode = n_gen / t_decode`，`L_TPOT = t_decode / n_gen`，`L_TTFT = t_prefill`。

### Host 补测
```bash
cargo build -p myos_runtime --release
unset MYOS_GEMV_PACK MYOS_GEMV_PREFETCH
./target/release/myos --arena-s1 \
  --weights /workspace/MyOS/weights/Qwen2.5-1.5B-Instruct/model.safetensors \
  --prompt-ids arena/prompts/s1_prompt.token_ids.txt \
  --ref-ids runs/s1-llamacpp-bf16-b11960/token_ids.txt \
  --steps 128 --threads 8 --warmup 3 --repeats 5 \
  --fingerprint-dir runs/s1-myos-bf16-t8-k10 \
  --cost-card arena/s1-myos-bf16-t8-k10-cost-card.md
```
