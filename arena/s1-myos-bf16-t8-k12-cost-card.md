# Cost Card · S1 · MyOS BF16 threads=8

### 设定
- fingerprint: `s1-myos-bf16-t8-k12`
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
- challenger commit: 6a0e5c8d7e9e912f859d9204f845e8b78598cd54（测时 tip `6a0e5c8`；**#53 已关，不合入**。该提交不在 main。）
- baseline: llama.cpp b11960 `s1-llamacpp-bf16-b11960` threads=8
- baseline source: `arena/e1-s1-bf16-cost-card.md` (T_decode 17.2348, L_TTFT 608.11 ms, L_TPOT 58.022 ms, M_peak 3050.953 MiB)
- knife-9 anchor: T_decode **15.4880**（`arena/s1-myos-bf16-t8-k9-cost-card.md`，main @ `1460ef22`）。本卡不替换该中位。
- cmdline: `./target/release/myos --arena-s1 --weights /workspace/MyOS/weights/Qwen2.5-1.5B-Instruct/model.safetensors --prompt-ids arena/prompts/s1_prompt.token_ids.txt --ref-ids runs/s1-llamacpp-bf16-b11960/token_ids.txt --steps 128 --threads 8 --warmup 3 --repeats 5 --fingerprint-dir runs/s1-myos-bf16-t8-k12 --cost-card arena/s1-myos-bf16-t8-k12-cost-card.md`

### 结果（中位数）
| metric | MyOS | llama.cpp | ratio (MyOS/base) | verdict |
|--------|------|-----------|-------------------|---------|
| T_decode (tok/s) | 13.5677 | 17.2348 | 0.7872 | L |
| L_TTFT / L_prefill (ms) | 1560.521 | 608.110 | 2.5662 | L |
| L_TPOT (ms) | 73.704 | 58.022 | 1.2703 | L |
| M_peak (MiB RSS) | 2962.938 | 3050.953 | 0.9712 | T |
| t_load (s, not in T_decode) | 0.0004 | — | — | appendix |
| T_e2e (tok/s) | 23.0922 | — | — | appendix |

### 正确性
- greedy exact match: PASS
- det consistency: PASS

### 代价拆解
- 算力：decode 仍是 GEMV 行块分给 8 个 worker。prefill 把整段 prompt 做成一次分块投影（同一权重行乘上 chunk 内每个 token）。每个 (token, 行) 仍是 `j=0..k-1` 先乘后加，非 FMA，无跨行归约，不 pack 权重。软件预取默认关（`MYOS_GEMV_PREFETCH=1` 才开）。
- 内存：`M_peak` 是加载完成后、prefill 与 decode 期间的 VmRSS 峰值。权重保持 BF16 视图，无整表 f32。
- 工程：数值核 no_std；`std` 线程池只做确定行分割。刀⑥ panel pack 默认关（`gemv_pack` 见上）。
- 功能债：协议 batch 仍为 1。本卡不含 S2/S3，不含 Q4。

### 结论一句
- 胜负：场景负（加权和 Some(-80)）。T_decode 0.35、L_TPOT 0.25、L_TTFT 0.20、M_peak 0.20；±5% 带宽。这不是「已经胜过 llama」的对外句，除非四项与场景分都是胜。
- 主要代价：行维并行对齐 llama `-t 8`；不把未测数字写成 ±5% 胜负。
- 硬门：host 中位已填。T_decode 13.5677 相对刀⑨ 15.4880 约 −12.4%，未达 ≥16.37 / 16.42，加权仍 −80，L_TTFT 1560.521 ≤ 2193.6 过，**硬门 FAIL**。KV 直读不进默认（**#53 已关，不合入**）。下一刀基线 = 刀⑨（T_decode 15.4880），禁止用 13.5677 / 13.57、11.5767 / 11.58 或 11.8074 / 11.81 当改善基线。
- 是否值得为 Phase2 继续加码：S1 闭环先落地。Phase2 与否看 host 填完本卡之后的场景分，不在本刀宣布。

### 备注
warmup ids match scored: true. Per-run dirs: warm-* discarded, score-* scored. Child `metrics.json` is one process; this directory's `metrics.json` is the W/R aggregate. 代码测点是测时 tip `6a0e5c8`，**#53 已关，不合入**。

### 计量边界（协议 §2.1）
1. `t_load` 单独计，不进入 L_prefill / T_decode。
2. `t_prefill`：从提交 prompt tokens 到产出第一个新 token 之前（含 KV / workspace 分配与整段 chunk prefill，不是 `n_prompt` 次 decode 同构 forward）。
3. `t_decode`：`greedy_decode` 跑完 `n_gen` 个新 token。`T_decode = n_gen / t_decode`，`L_TPOT = t_decode / n_gen`，`L_TTFT = t_prefill`。
