# Phase1 刀⑦ · 代码落地说明（S1 对打闭环）

| 项 | 内容 |
|----|------|
| 口径 | `docs/phase1-next-coarse-knife-7.md` |
| 基线 | main `5bb4065f`（刀⑥ pack 默认关） |
| 本环境 | 无 Qwen2.5-1.5B 权重，**没有**跑 S1 128/128。CORRECT / DET / W=3 R=5 中位标 **UNMEASURED**。host 用下面的命令重跑后再把数字写进主表。 |

## 已落地

1. **threads=8** 是 showdown 默认。`--threads N` 写入指纹。`batch` 仍是 1。
2. GEMV 按 [`gemv_row_spans`] 把输出行切成连续块。每块调用原来的 no_std 核（先乘后加，非 FMA）。没有跨行归约。`std` 线程池只负责这块分割；`tensor_core` 不开 `std` feature 时仍是单线程核。
3. 单次 `--weights` 写 `metrics.json`：`t_load` / `t_prefill` / `t_decode` / `M_peak`。`t_load` 不进 L_prefill 与 T_decode。
4. `--arena-s1` 父进程拉起 W 次丢弃 + R 次计分（默认 3 和 5），写 `runs/s1-myos-bf16-t8/` 与 `arena/s1-myos-bf16-cost-card.md`。子进程才解码。
5. `MYOS_GEMV_PACK` 仍默认关。指纹字段 `gemv_pack`。pack 打开或 threads≠8 时场景判 **incomparable**（协议 §1.3），不拿去和 `-t 8` 记 W/T/L。

## 本 VM 量到的（不是 S1 墙钟）

release 微基准，gate/up 形状 `[8960, 1536]`，同一进程先 threads=1 再 threads=8，比特一致。`nproc` / `available_parallelism` = **4**。

| | per call | 4 次调用的一段 |
|--|----------|----------------|
| threads=1 | 0.001486 s | 0.005943 s |
| threads=8 | 0.000411 s | 0.001646 s |
| 比值 | **3.611×** | 同左 |

这是软记录，不是 ≥2× 硬门，也不是 S1，更不是对 llama 的 ±5% 胜负。刀⑥单线程 S1 墙钟约 80 s（`-t 1` 指纹）；llama S1 BF16 中位约 8.8 s（`-t 8`）。host 要用 **threads=8** 重跑才能填主表。

串行核相对 naive 的刀⑤地板仍在（本机 release：gate_up 约 4.05×，down 约 5.07×）。pack 路径仍不是默认，微基准里相对 strided 没有变成主路径。

## host 重跑

权重与金标准 token 沿用 E1/E2 路径。不要设 `MYOS_GEMV_PACK`。

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

产物：`runs/s1-myos-bf16-t8/metrics.json`（中位）、`det_evidence.json`、`token_ids.txt`、`tokens.txt`（id 序列，MyOS 不 detokenize）、`cost-card.md`。CORRECT 或 DET 失败时进程退出码 1，卡上写场景负且吞吐作废。

可选对照（不入主表）：同一命令加 `--threads 1`，输出目录会是 `runs/s1-myos-bf16-t1`。该指纹与 llama `-t 8` **不可比**。
