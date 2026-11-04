# 刀⑩未落地 + 刀⑪口径（2026-11-04）

main 停在刀⑨ `1460ef22`。刀⑩没有合入这条历史。本刀的对照基线是刀⑨ host 中位 **T_decode 15.4880**（指纹 `s1-myos-bf16-t8-k9`）。不另立一条 decode 中位当基线。

## ① 刀⑪改了哪条粗路径

一口，默认开，没有加速开关（没有 `MYOS_FUSE` 一类回退）。pack 仍默认关，软件预取仍默认关（`MYOS_GEMV_PREFETCH=1` 才开）。行宽和寄存器块宽不动。乘加序不动：每行仍是 `j = 0..k-1` 先乘后加，非 FMA，无跨行归约。

1. **多投影融合。** Decode 的 Q、K、V 共用同一次激活，一次 worker 波算完再加 bias。gate 与 up 同样：一次波，然后 SiLU，down 仍是下一次 GEMV（它读的是 `silu(gate)*up`）。每个 worker 先走自己那一段第一张矩阵的行，再走下一张，中间不 join。数值核仍是刀⑨那份按行的 BF16→f32 点积。
2. **跨 decode 步权重视图借用。** `greedy_decode` 在完整权重上把每一层的 BF16 视图借一次，后面的 step 复用这些切片，不再逐步重查名字。字节仍按步从映射里流过，不复制成 f32，也不 pack。缺张量时退回逐层查找，以便较早的层仍能按原错误路径写入 KV。

Prefill 保持刀⑧的整段 chunk（`gemm_bf16`），不走这条 decode 波。

## ② 硬门（host 补测；本环境无 1.5B 权重）

| 硬门 | 门槛 |
|------|------|
| CORRECT / DET | PASS @ threads=8 |
| T_decode | ≥16.42，或（≥16.37 且场景加权严格优于 −80） |
| L_TTFT | ≤2193.6 |
| 相对刀⑨ | T_decode ≥ **15.4880** |

指纹目录：`runs/s1-myos-bf16-t8-k11/`。账本草稿：`arena/s1-myos-bf16-t8-k11-cost-card.md`。端到端 S1 在写这份草稿的环境里是 **UNMEASURED**。上表没有填实测中位。

## ③ 微基准（不是 S1）

`cargo test -p tensor_core --release release_same_x`，本机 release，threads=8，权重反复打在同一块缓冲上（热缓存，不是 mmap 冷流）。两次：

| 组 | 形状 | separate / fused |
|----|------|------------------|
| Q/K/V | `[1536,1536]` + `[256,1536]` + `[256,1536]` | 1.185，1.164 |
| gate/up | 两个 `[8960,1536]` | 1.132，1.137 |

比值大于 1 表示融合波更短。这组数不能乘到 15.4880 上当成端到端 tok/s。S1 的权重大约 2.8GiB，不会整段留在缓存里。

## ④ host 补测

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
