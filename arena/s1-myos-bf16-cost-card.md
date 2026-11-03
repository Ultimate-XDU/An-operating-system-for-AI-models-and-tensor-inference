# Cost Card · S1 · MyOS BF16 threads=8

### 设定
- fingerprint: `s1-myos-bf16-t8`
- model: Qwen2.5-1.5B-Instruct
- track: SafeTensors BF16 (main track)
- device: cpu, n_threads: 8
- batch: 1
- quant_or_dtype: bf16
- acc_path: bf16_w_fp32_acc
- gemv_pack: false（刀⑥ panel pack 不是默认路径；`MYOS_GEMV_PACK` 未设）
- deterministic: ON (greedy argmax). seed: null (no RNG)
- n_ctx=256, n_prompt=128, n_gen=128
- protocol: W=3 discard, R=5 median
- challenger commit: 本刀合入后的 git HEAD（本环境未跑权重）
- baseline: llama.cpp b11960 `s1-llamacpp-bf16-b11960` threads=8
- baseline source: `arena/e1-s1-bf16-cost-card.md` (T_decode 17.2348, L_TTFT 608.110 ms, L_TPOT 58.022 ms, M_peak 3050.953 MiB)
- measurement: **UNMEASURED** on the authoring VM (no 1.5B SafeTensors). Host rerun overwrites this file.
- cmdline (host):

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
  --fingerprint-dir runs/s1-myos-bf16-t8 \
  --cost-card arena/s1-myos-bf16-cost-card.md
```

### 结果（中位数）
| metric | MyOS | llama.cpp | ratio (MyOS/base) | verdict |
|--------|------|-----------|-------------------|---------|
| T_decode (tok/s) | UNMEASURED | 17.2348 | — | UNMEASURED |
| L_TTFT / L_prefill (ms) | UNMEASURED | 608.110 | — | UNMEASURED |
| L_TPOT (ms) | UNMEASURED | 58.022 | — | UNMEASURED |
| M_peak (MiB RSS) | UNMEASURED | 3050.953 | — | UNMEASURED |
| t_load (s, not in T_decode) | UNMEASURED | — | — | appendix |
| T_e2e (tok/s) | UNMEASURED | 31.8608 | — | appendix |

llama 列是 E1 已测中位，用来做 ratio 的分母。MyOS 分子待 host 写入。在那之前 **不计算 W/T/L**。

### 正确性
- greedy exact match: UNMEASURED（host 对 `runs/s1-llamacpp-bf16-b11960/token_ids.txt` 128/128）
- det consistency: UNMEASURED（同指纹 R=5 id 序列）

### 代价拆解
- 算力：GEMV 行块分给 8 个 worker；每行内部仍是 `j=0..k-1` 先乘后加，非 FMA，与 scalar `dot_row` 同比特。无跨行归约。
- 内存：`M_peak` 定义为加载完成后、prefill 与 decode 期间的 VmRSS 峰值。权重保持 BF16 视图，无整表 f32。
- 工程：数值核 no_std；`std` 线程池只做确定行分割。刀⑥ panel pack 默认关。
- 功能债：batch 仍为 1。本卡不含 S2/S3，不含 Q4。
- 本 VM 软记录（**不是** S1）：gate/up `[8960, 1536]` release，threads=1 → threads=8 为 **3.611×**（0.001486 s → 0.000411 s / call，4 cores）。见 `docs/phase1-knife7-closeout.md`。

### 结论一句
- 胜负：不可比。本机没有 1.5B 权重，MyOS 中位是 UNMEASURED。llama 分母已在表里，host 重跑后按协议 §4.2 填 W/T/L（T_decode 0.35、L_TPOT 0.25、L_TTFT 0.20、M_peak 0.20，±5%）。
- 主要代价：行维并行对齐 llama `-t 8`；不把未测数字写成 ±5% 胜负。
- 是否值得为 Phase2 继续加码：S1 闭环先落地。Phase2 与否看 host 填完本卡之后的场景分，不在本刀宣布。

### 计量边界（协议 §2.1）
1. `t_load` 单独计，不进入 L_prefill / T_decode。
2. `t_prefill`：从提交 prompt tokens 到产出第一个新 token 之前（含 KV / workspace 分配与 `n_prompt` 次 forward）。
3. `t_decode`：`greedy_decode` 跑完 `n_gen` 个新 token。`T_decode = n_gen / t_decode`，`L_TPOT = t_decode / n_gen`，`L_TTFT = t_prefill`。
