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

## Qwen2.5 weight keys

`model_core::qwen25_layout` names every Qwen2.5-1.5B-Instruct tensor (`model.embed_tokens.weight`, `model.layers.{i}.*` for `i = 0..27`, and `model.norm.weight`) and the shapes from `docs/qwen25-1.5b-instruct-layout.md`. There is no `lm_head` key (`tie_word_embeddings`). `model_core::qwen25_bf16` borrows those payloads as little-endian BF16 byte views from any `WeightMap`, including `SafeTensorsWeights`, and rejects a missing name, a non-BF16 dtype, or a shape that is not the layout shape. It does not decode the bytes to `f32`. `QwenStub` still uses the f32 cycle fixture.

## BF16 numeric path

Phase1 main-track weights stay BF16 byte views. `tensor_core::gemv_bf16` is a row-major little-endian BF16 matrix times an f32 vector, accumulated in f32. `tensor_core::embed_gather_bf16` copies one `[vocab, hidden]` row into an f32 vector. Both decode BF16 (the top 16 bits of f32) inside the op. They do not call `load_f32_tensor`, and `load_f32_tensor` stays F32-only.

Fingerprint intent, hash later: `quant_or_dtype=bf16`, `acc_path=bf16_w_fp32_acc`.

`gemv_f32` and the `cycle_fixture` smoke stay on the f32 fixture. `cargo run -p myos_runtime` is still `[1, 2, 3, 0]`. Unit tests build tiny BF16 tables in process; they do not download Qwen.

## RMSNorm

`tensor_core::rmsnorm_bf16_weight` is the next main-track op. Activations stay f32 across the op. Gamma is a little-endian BF16 vector (`2 * n` bytes), promoted with `bf16_le_to_f32` inside the loop. `rmsnorm_f32` is the same kernel with an f32 gamma, for exact comparisons.

`Σx²` is one thread, left to right: `sum += x[i] * x[i]` (multiply, then add; not `mul_add`, not a tree). Then `inv_rms = 1 / sqrtf(sum / n + eps)` (IEEE-754 sqrt, not an approximate rsqrt) and `y[i] = w[i] * (x[i] * inv_rms)`. That reduction is the exact-sensitive part of `acc_path=bf16_w_fp32_acc` for this op: BF16 weights, f32 accumulation, no second dtype and no parallel sum. Qwen2.5 passes `RMS_NORM_EPS_QWEN25` (`1e-6`, `config.json` `rms_norm_eps`). Length mismatches return `Err` and leave `y` unchanged. The smoke line is still `[1, 2, 3, 0]`.

## RoPE

`tensor_core::rope_f32` and `rope_f32_inplace` rotate one head or a batch of heads that share one position. Activations stay f32. Qwen2.5 passes `ROPE_THETA_QWEN25` (`1e6`). Qwen2.5-1.5B passes `HEAD_DIM_QWEN25_1_5B` (`128` = 1536 / 12). `rope_inv_freqs` writes the pair table those rotates use.

Pairing is NeoX / half-rotate, the same layout as HF Qwen2 `rotate_half` and llama.cpp `LLAMA_ROPE_TYPE_NEOX`. With `h = head_dim / 2`, lane `i` rotates with lane `i + h`: `y[i] = x[i] cos - x[i+h] sin`, `y[i+h] = x[i+h] cos + x[i] sin` (multiply then add or subtract, no `mul_add`). The same angle is used on both halves. Adjacent even/odd lanes are not a second entry point.

`inv_freq[i] = 1 / powf(theta, (2i) / head_dim)`, then `angle = f32(position) * inv_freq[i]` and libm `cosf` / `sinf`. Each pair is independent (no iterative `freq *= theta**(-2/d)`). Position 0 is the identity on finite inputs. An odd `head_dim`, an empty buffer, or a length that is not a positive multiple of `head_dim` returns `Err` and does not write. The smoke line is still `[1, 2, 3, 0]`.

## GQA attention

`tensor_core::gqa_attn_f32` is one query position against cached K and V. Activations stay f32 (Q, K, V, scores, probabilities, output). There is no weight byte view: Q/K/V projections stay on `gemv_bf16` later. Counts sit on `GqaShape`. Qwen2.5-1.5B passes `NUM_ATTENTION_HEADS_QWEN25_1_5B` (12), `NUM_KV_HEADS_QWEN25_1_5B` (2), and `HEAD_DIM_QWEN25_1_5B` (128), so `GQA_Q_PER_KV_QWEN25_1_5B` is 6. Query head `h` reads KV head `h / 6` (HF `repeat_kv`, contiguous groups).

Layout, no batch axis: Q and Y are `[num_q_heads, head_dim]`. K and V are `[num_kv_heads, seq_len, head_dim]`. `scale = 1 / sqrtf(head_dim)` once (`attn_scale_f32`), then each score is `dot * scale` after a left-to-right dot (multiply, then add; not `mul_add`). Softmax is one thread over the visible keys: max starts at key 0 (`>`), `expf(score - max)`, left-to-right sum, then `e / sum` (division, not a reciprocal multiply). The value mix is the same left-to-right order. `probs` is scratch of length `seq_len`; on success it holds the last query head.

`GqaMask::Full` scores the whole cache (decode that already stored only the legal prefix). `GqaMask::Causal { query_pos }` drops key index `t > query_pos` before the dot, so those V slots are not read. `query_pos >= seq_len - 1` keeps every key. A zero count, a query-head count that is not a multiple of the KV-head count, or a bad buffer length returns `Err` and does not write. KV-cache allocation and a multi-query prefill kernel are not in this step. The smoke line is still `[1, 2, 3, 0]`.

## SwiGLU

`tensor_core::swiglu_bf16` is one token's Qwen2 MLP. Activations stay f32. `mlp.gate_proj.weight`, `mlp.up_proj.weight`, and `mlp.down_proj.weight` stay little-endian BF16 views and go through `gemv_bf16` (row-major `[out, in]`, f32 dot, left to right, multiply then add). `swiglu_f32` is the same graph on f32 weights. Qwen2.5-1.5B passes `HIDDEN_SIZE_QWEN25_1_5B` (1536) and `INTERMEDIATE_SIZE_QWEN25_1_5B` (8960): gate and up are `[8960, 1536]`, down is `[1536, 8960]`. Counts sit on `SwigluShape` so a test can shrink them.

`gate = gemv(gate_proj, x)`, `up = gemv(up_proj, x)`, `h[i] = silu(gate[i]) * up[i]`, `y = gemv(down_proj, h)`. None of the three projections adds a bias (Qwen2.5 `mlp.*` has none). `silu_f32` is `z / (1 + expf(-z))` (libm `expf`, one division). That is llama.cpp `ggml_silu_f32`. It is not `z * (1 / (1 + expf(-z)))`; those differ by an ulp. The product is one multiply after SiLU returns, not `mul_add`. `gate`, `up`, and `h` are caller scratch of length `intermediate`. On success they hold those three vectors. A zero dimension or a bad buffer length returns `SwigluError::Shape` and does not write. The transformer block is not this step. The smoke line is still `[1, 2, 3, 0]`.

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
