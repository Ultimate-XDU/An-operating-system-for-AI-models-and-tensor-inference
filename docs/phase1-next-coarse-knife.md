# Phase1 刀④口径（主推已收口）

| 项 | 内容 |
|----|------|
| 文档 | `docs/phase1-next-coarse-knife.md` |
| 状态 | 刀④按主推落地。细粒度新算子不在本刀 |
| 效力 | 与 `docs/phase1-ember-port-must-rewrite.md` 冲突时以三门闩为准；与协议冲突时以 `PHASE1-ARENA-PROTOCOL-v0.1.md` 为准 |

## 主推

`--weights PATH`：mmap SafeTensors → 已有 `forward_token` / `prefill` / `greedy_decode`，形状用 `BlockShape::qwen25_1_5b()`。未给 `--prompt-ids` 时 prompt token id 是 `0`，默认 4 步（`--steps N`）。成功行是 `MyOS Phase1 weights decode (prompt=0, steps=N): [...]`。同时写 `runs/s1-weights/run_fingerprint.json`。

三门闩：

1. **BF16**：权重保持字节视图；升精只在已有算子内。指纹 `quant_or_dtype=bf16`，`acc_path=bf16_w_fp32_acc`。禁止加载期整表 `Vec<f32>`，禁止在映射胶水里 decode。
2. **exact**：不引入 rayon，不换 NeoX / QKV bias / SwiGLU / 串行核。本刀不要求已过 vs llama.cpp exact。
3. **no_std**：控制流与键映射留在 `model_core`。mmap、CLI、指纹留在 `myos_runtime`。core 不加 anyhow / rayon / tokenizers / onig。

默认无参数仍是 cycle `[1, 2, 3, 0]`。`--block-fixture` 仍是进程内微型块。测试不下载 2.88GiB 文件；同路径的自洽用微型 SafeTensors 在 `fixture_shape` 上跑两次。CLI 对那份微型文件必须因 1.5B 形状不符而失败，不能当成夹具解码成功。

指纹里填得了的 §1.3 字段照实写。`model_sha256` 与 `seed` 为 null。`scenario_id` 是占位 `S1`。

E2 宿主 CLI：`--prompt-ids` 在 `--weights` 上预填整段 token id，`max_seq = prompt.len() + steps`；`--ref-ids` 打印 `CORRECT`，不匹配仍退出 0。未给 `--prompt-ids` 时仍是单个 id 0。

## 未采用

| 备选 | 不采用的原因 |
|------|----------------|
| `--weights` 继续只打印张量名，然后走 cycle | 主路仍然没吃到真权重 |
| 加载期整表 BF16→f32 | 违反 BF16 门闩，也放大常驻 |
| 把 mmap 或 CLI 放进 `model_core` | 违反 no_std 门闩 |
| 为本刀新写 GEMV / attention / tokenizer | 刀④只接线，不新开算子 |
| 把布局文档里的 LFS SHA256 写进每次指纹 | 那是特定文件的哈希，不是这次进程算出来的 |
| 本刀对打 llama.cpp exact、Q4、unikernel | 不在本刀验收里 |
