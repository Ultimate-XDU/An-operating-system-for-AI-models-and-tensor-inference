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
- challenger commit: null（host 重跑后写入）
- baseline: llama.cpp b11960 `s1-llamacpp-bf16-b11960` threads=8
- baseline source: `arena/e1-s1-bf16-cost-card.md` (T_decode 17.2348, L_TTFT 608.11 ms, L_TPOT 58.022 ms, M_peak 3050.953 MiB)
- cmdline: `./target/release/myos --arena-s1 --weights /workspace/MyOS/weights/Qwen2.5-1.5B-Instruct/model.safetensors --prompt-ids arena/prompts/s1_prompt.token_ids.txt --ref-ids runs/s1-llamacpp-bf16-b11960/token_ids.txt --steps 128 --threads 8 --warmup 3 --repeats 5 --fingerprint-dir runs/s1-myos-bf16-t8-k12 --cost-card arena/s1-myos-bf16-t8-k12-cost-card.md`
- measurement: **UNMEASURED** on this machine (no 1.5B weight file in the authoring environment). The harness above is what the host runs. It overwrites this card.

### 结果（中位数）
| metric | MyOS | llama.cpp | ratio (MyOS/base) | verdict |
|--------|------|-----------|-------------------|---------|
| T_decode (tok/s) | UNMEASURED | 17.2348 | UNMEASURED | UNMEASURED |
| L_TTFT / L_prefill (ms) | UNMEASURED | 608.110 | UNMEASURED | UNMEASURED |
| L_TPOT (ms) | UNMEASURED | 58.022 | UNMEASURED | UNMEASURED |
| M_peak (MiB RSS) | UNMEASURED | 3050.953 | UNMEASURED | UNMEASURED |
| t_load (s, not in T_decode) | UNMEASURED | — | — | appendix |
| T_e2e (tok/s) | UNMEASURED | — | — | appendix |

### 正确性
- greedy exact match: UNMEASURED
- det consistency: UNMEASURED

### 代价拆解
- 算力：decode 仍是 GEMV 行块分给 8 个 worker。prefill 把整段 prompt 做成一次分块投影（同一权重行乘上 chunk 内每个 token）。每个 (token, 行) 仍是 `j=0..k-1` 先乘后加，非 FMA，无跨行归约，不 pack 权重。软件预取默认关（`MYOS_GEMV_PREFETCH=1` 才开）。注意力不再每步打包整段 KV 前缀：cache 按 `max_seq` 步长直接读，当前 token 附在旁边。GQA 的点积与每条 value lane 仍从左到右先乘后加。
- 内存：`M_peak` 是加载完成后、prefill 与 decode 期间的 VmRSS 峰值。权重保持 BF16 视图，无整表 f32。decode / prefill 去掉了整段 prefix pack 缓冲。
- 工程：数值核 no_std；`std` 线程池只做确定行分割。刀⑥ panel pack 默认关（`gemv_pack` 见上）。
- 功能债：协议 batch 仍为 1。本卡不含 S2/S3，不含 Q4。

### 结论一句
- 胜负：不可比。本机没有 1.5B 权重，MyOS 中位是 UNMEASURED。llama 分母已写在表里，host 重跑后按 §4.2 填 W/T/L。
- 主要代价：行维并行对齐 llama `-t 8`；不把未测数字写成 ±5% 胜负。
- 是否值得为 Phase2 继续加码：S1 闭环先落地。Phase2 与否看 host 填完本卡之后的场景分，不在本刀宣布。

### 备注
刀⑫草稿。基线是刀⑨ `T_decode` 15.4880（`s1-myos-bf16-t8-k9`）。禁止用刀⑪ 11.5767 或刀⑩ 11.8074 当改善基线。硬门：CORRECT+DET PASS @ threads=8；T_decode ≥ 16.42，或（≥ 16.37 且场景加权严格优于 −80）；L_TTFT ≤ 2193.6；相对刀⑨ T_decode ≥ 15.4880。口径：`docs/phase1-knife11-verdict-and-knife12.md`。云端没有跑 S1，表内空位不是测量结果。

### 计量边界（协议 §2.1）
1. `t_load` 单独计，不进入 L_prefill / T_decode。
2. `t_prefill`：从提交 prompt tokens 到产出第一个新 token 之前（含 KV / workspace 分配与整段 chunk prefill，不是 `n_prompt` 次 decode 同构 forward）。
3. `t_decode`：`greedy_decode` 跑完 `n_gen` 个新 token。`T_decode = n_gen / t_decode`，`L_TPOT = t_decode / n_gen`，`L_TTFT = t_prefill`。
