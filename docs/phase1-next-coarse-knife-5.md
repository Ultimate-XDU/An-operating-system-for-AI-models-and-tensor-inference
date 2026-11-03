# Phase1 刀⑤口径（E2 PASS 后下一粗刀）

| 项 | 内容 |
|----|------|
| 文档 | `docs/phase1-next-coarse-knife-5.md` |
| 前提 | E2 S1 CORRECT PASS @ `42acbb39`（`runs/e2-myos-s1`，128/128 vs llama.cpp BF16） |
| 效力 | 与协议冲突时以 `PHASE1-ARENA-PROTOCOL-v0.1.md` 为准；副赛道 Q4 仍不开 |

## 主推：刀⑤ · S1 主路径吞吐粗块

**选它而不是先开 S2 的理由（可核对）**

- 协议 A7：正确性过后才谈吞吐胜负；E2 已收口，scoreboard 下一格是「主赛道 S1 吞吐对打」。
- 量级：llama.cpp S1 BF16 中位 T_decode ≈ **17.23 tok/s**、墙钟约 **8.8 s**（`-t 8`，`arena/e1-s1-bf16-baseline.md`）；MyOS 同场景墙钟 ≈ **31 min**、`threads=1`、naive `gemv_bf16`（`arena/e2-s1-myos-exact.md`）。未缩差距前开 S2（`n_prompt=2048`）只会放大墙钟，难迭代 exact/对打。
- Phase1 宣称胜要 S1∩S2 场景胜；瓶颈在 decode/GEMV 热路径，不是缺场景脚手架。

**刀范围（粗块，非整仓 fork，非细算子清单）**

- 热路径：`gemv_bf16`（及由其主导的 decode/prefill 墙钟）——允许 cache-blocked / SIMD（如目标机 AVX2）等 **同一数值语义** 的实现替换。
- 保持：`no_std` core、BF16 字节视图、算子内 fp32 累加、`acc_path=bf16_w_fp32_acc`、NeoX / QKV bias / SwiGLU / 串行确定性主路径。
- 禁止本刀：加载期整表 f32、引入 rayon 破坏 deterministic exact、换权重格式、开 Q4 对打、为 S2/S3 新场景矩阵铺量。

**验收口径**

1. **exact 不破**：同 E2 配对下 S1 `CORRECT: PASS`（`--ref-ids` vs `runs/s1-llamacpp-bf16-b11960/token_ids.txt`，128/128）。
2. **指纹不变门闩**：`quant_or_dtype=bf16`，`acc_path=bf16_w_fp32_acc`；权重仍为 SafeTensors mmap 视图。
3. **吞吐门槛（相对刀④/E2 基线，同机可复现）**：S1（128+128）墙钟相对 E2 ≈31 min 基线 **≥10×** 缩短（即墙钟 ≲ 3.1 min @ 与基线相同的 `threads` 设定）；并写出 `T_decode` / `L_TPOT`（可先单次，正式 W=3/R=5 cost-card 可另刀）。不要求本刀已胜 llama ±5%。
4. **no_std**：SIMD/blocked 核留在 `tensor_core`（或等价 no_std 核）；mmap/CLI/指纹仍在 runtime。
5. **回归**：默认 cycle smoke 与微型 ST 自洽测试仍绿。

**CloudAgent**：适合开干（热路径改写 + 门闩复跑）。须钉：不得改 exact 语义；PR 附 S1 CORRECT 与墙钟前后对比。

## 备选：刀⑤b · S2 exact 门闩

- 仅当刀⑤吞吐已过门槛、或月星/OSdeveloper 显式要先扩场景。
- 验收：S2 prompt token 文件 + MyOS greedy exact vs 预先备好的 llama S2 ref；`n_ctx` 对齐协议；CORRECT PASS。
- **现在不主推**：naive 路径下 prefill 约 16× S1，机器时间不划算。

## 明确不做（本决策）

| 项 | 原因 |
|----|------|
| 先开 S2/S3 场景矩阵 | 墙钟未收敛，对打无意义 |
| 副赛道 Q4 对打 | OSdeveloper 约束：先不开 |
| 细算子逐个微优化清单 | 本刀是粗块热路径，不是算子超市 |
| 本刀宣称胜过 llama.cpp | 只收 exact + 量级跃迁；正式 cost-card 对打另派 |
