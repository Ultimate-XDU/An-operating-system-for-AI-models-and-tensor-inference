# An operating system for large models and tensor inference

AI-Native Runtime：面向大模型与张量推理的极简 Unikernel。

## Working directory

本地工作目录：`/workspace/MyOS`（与本仓库对齐）。

## Phase 1 focus

- WeightLoaderCore（SafeTensors / GGUF 零拷贝加载）
- ModelCore（类型安全模型抽象）
- 最小推理循环（prefill + decode，CPU）
- 对打协议：见 `PHASE1-ARENA-PROTOCOL-v0.1.md`

## Phase 1 P0 smoke

```
cargo test --workspace
cargo run -p myos_runtime
```

`cargo run -p myos_runtime` prints `MyOS Phase1 smoke decode (prompt=0): [1, 2, 3, 0]`.

## SafeTensors loader

`cargo test -p weight_loader` covers `SafeTensorsWeights::parse` with a tiny in-process fixture (no model download). The `safetensors` crate is built with default features off, so its `std` feature stays disabled. Tensor bytes are borrowed from the caller's buffer. Mapped storage dtypes are `F32`, `F16`, and `BF16`.

The runtime smoke above is unchanged: it still loads `MmapWeights` via `cycle_fixture`, not a SafeTensors file.

## Team

GitHub 写操作仅由 OSdeveloper 负责；博弈手 / 仓寻 / 深析 / 联调官负责方案、搜索、分析与联调结论。
