# An operating system for large models and tensor inference

AI-Native Runtime：面向大模型与张量推理的极简 Unikernel。

## Working directory

本地工作目录：`/workspace/MyOS`（与本仓库对齐）。

## Phase 1 focus

- WeightLoaderCore（SafeTensors / GGUF 零拷贝加载）
- ModelCore（类型安全模型抽象）
- 最小推理循环（prefill + decode，CPU）
- 对打协议：见 `PHASE1-ARENA-PROTOCOL-v0.1.md`
- 布局钉死：见 `docs/qwen25-1.5b-instruct-layout.md`
- BF16 计算策略：见 `docs/phase1-bf16-compute-strategy.md`

## Phase 1 P0 smoke

```
cargo test --workspace
cargo run -p myos_runtime
```

`cargo run -p myos_runtime` prints `MyOS Phase1 smoke decode (prompt=0): [1, 2, 3, 0]`.

Optional debug print of a SafeTensors file (tensor count and sorted names), then the same smoke line:

```
cargo run -p myos_runtime -- --weights path/to/file.safetensors
MYOS_WEIGHTS=path/to/file.safetensors cargo run -p myos_runtime
```

`--weights` wins when both are set. The smoke path does not load that file. A bad path exits before the smoke line.

## SafeTensors loader

`cargo test -p weight_loader` covers `SafeTensorsWeights::parse` with a tiny in-process fixture (no model download). The `safetensors` crate is built with default features off, so its `std` feature stays disabled. Tensor bytes are borrowed from the caller's buffer. Mapped storage dtypes are `F32`, `F16`, and `BF16`.

The runtime smoke above is unchanged: it still loads `MmapWeights` via `cycle_fixture`, not a SafeTensors file.

## BF16 numeric path

Phase1 main-track weights stay BF16 byte views. `tensor_core::gemv_bf16` is a row-major little-endian BF16 matrix times an f32 vector, accumulated in f32. `tensor_core::embed_gather_bf16` copies one `[vocab, hidden]` row into an f32 vector. Both decode BF16 (the top 16 bits of f32) inside the op. They do not call `load_f32_tensor`, and `load_f32_tensor` stays F32-only.

Fingerprint intent, hash later: `quant_or_dtype=bf16`, `acc_path=bf16_w_fp32_acc`.

`gemv_f32` and the `cycle_fixture` smoke stay on the f32 fixture. `cargo run -p myos_runtime` is still `[1, 2, 3, 0]`. Unit tests build tiny BF16 tables in process; they do not download Qwen.

## RMSNorm

`tensor_core::rmsnorm_bf16_weight` is the next main-track op. Activations stay f32 across the op. Gamma is a little-endian BF16 vector (`2 * n` bytes), promoted with `bf16_le_to_f32` inside the loop. `rmsnorm_f32` is the same kernel with an f32 gamma, for exact comparisons.

`Σx²` is one thread, left to right: `sum += x[i] * x[i]` (multiply, then add; not `mul_add`, not a tree). Then `inv_rms = 1 / sqrtf(sum / n + eps)` (IEEE-754 sqrt, not an approximate rsqrt) and `y[i] = w[i] * (x[i] * inv_rms)`. That reduction is the exact-sensitive part of `acc_path=bf16_w_fp32_acc` for this op: BF16 weights, f32 accumulation, no second dtype and no parallel sum. Qwen2.5 passes `RMS_NORM_EPS_QWEN25` (`1e-6`, `config.json` `rms_norm_eps`). Length mismatches return `Err` and leave `y` unchanged.

RoPE, GQA, and SwiGLU are not in this step. The smoke line is still `[1, 2, 3, 0]`.

## Feature flags

`weight_loader` is `no_std` + `alloc` by default. `tensor_core` and `model_core` depend on that default and do not enable host I/O.

| Feature | When | What it adds |
| --- | --- | --- |
| (default) | `no_std` library builds | `SafeTensorsWeights::parse(&[u8])` over a caller-owned buffer. No OS, no mmap. |
| `host` | non-test builds that open files (`myos_runtime` enables it) | `std` and `memmap2`. `SafeTensorsMmap::open(path)` maps a `.safetensors` file read-only. `parse()` borrows tensor payloads from the map; it does not copy them into a `Vec<u8>`. |

`myos_runtime` enables `weight_loader`'s `host` feature so the debug flag above can open a file. Do not turn `host` on for a `no_std` target: it links `std`. `cargo test -p weight_loader` still compiles the mmap module for that crate's own tests; `--features host` is what dependents need outside tests.

Tests generate a few-hundred-byte fixture at runtime (same `safetensors::serialize` path as the in-memory parser tests). Real `*.safetensors` weights stay gitignored.

## Team

GitHub 写操作仅由 OSdeveloper 负责；博弈手 / 仓寻 / 深析 / 联调官负责方案、搜索、分析与联调结论。
