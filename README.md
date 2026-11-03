# An operating system for large models and tensor inference

AI-Native Runtime：面向大模型与张量推理的极简 Unikernel。

## Working directory

本地工作目录：`/workspace/MyOS`（与本仓库对齐）。

## Phase 1 focus

- WeightLoaderCore（SafeTensors / GGUF 零拷贝加载）
- ModelCore（类型安全模型抽象）
- 最小推理循环（prefill + decode，CPU）
- 对打协议：见 `PHASE1-ARENA-PROTOCOL-v0.1.md`
- E2 S1 CORRECT PASS：实现笔记 `arena/e2-s1-myos-exact.md`；正式账本 `arena/e2-s1-correct.md`；总览 `arena/phase1-scoreboard.md`
- 布局钉死：见 `docs/qwen25-1.5b-instruct-layout.md`
- BF16 计算策略：见 `docs/phase1-bf16-compute-strategy.md`

## Phase 1 P0 smoke

```
cargo test --workspace
cargo run -p myos_runtime
```

`cargo run -p myos_runtime` prints `MyOS Phase1 smoke decode (prompt=0): [1, 2, 3, 0]`. That path does not read a weight file and does not write a fingerprint. `--steps N` changes how many tokens that cycle emits (default 4).

`--block-fixture` is a separate path (knife ③). It does not print that cycle line. It greedy-decodes a tiny in-process BF16 block and prints `MyOS Phase1 block fixture (prompt=0): [1, 1, 1, 1]`. It does not download Qwen. It writes `runs/s1-block-fixture/run_fingerprint.json`.

`--weights PATH` (knife ④) maps that SafeTensors file and greedy-decodes it at `BlockShape::qwen25_1_5b` (28 layers, hidden 1536, vocab 151936). Without `--prompt-ids` the prompt is the single token id 0. The default step count is 4. Success prints `MyOS Phase1 weights decode (prompt=0, steps=4): [...]` and writes `runs/s1-weights/run_fingerprint.json`. This binary does not download weights. A missing file, a file that is not SafeTensors, or a file whose tensors are not that BF16 layout exits before any decode line and does not fall through to the cycle.

`--threads N` is the GEMV worker count (default **8**, the llama S1 BF16 `-t 8` fingerprint). `1..=64`. The fingerprint records that number. `batch` stays 1. Rows are split deterministically; each row is still a left-to-right multiply-then-add. `--threads 1` is the old single-thread kernel and is not a showdown against `-t 8`.

`--prompt-ids PATH` (with `--weights`) reads whitespace-separated `u32` prompt ids, prefills all of them, then greedy-decodes `--steps` tokens. `max_seq` is `prompt.len() + steps`. Protocol S1 is `--steps 128` on a 128-id file. The fingerprint records that `n_prompt` and `n_gen`, and the run dir also gets `token_ids.txt` (space-separated generated ids) plus `metrics.json` (`t_load`, `t_prefill`, `t_decode`, `M_peak`). An empty file is an error. `--ref-ids PATH` compares those generated ids and prints `CORRECT: PASS` or `CORRECT: FAIL first_diff@i: got=X ref=Y` (length mismatch is its own FAIL line). A mismatch still exits 0.

`--arena-s1` (with `--weights`, `--prompt-ids`, and `--ref-ids`) is the S1 showdown parent. It does not decode itself. It spawns this binary `W + R` times (default `--warmup 3 --repeats 5`), discards the warmup, and writes medians to `runs/s1-myos-bf16-t<N>/` and `arena/s1-myos-bf16-cost-card.md` when `N` is 8. `--steps` defaults to 128 in this mode. `MYOS_GEMV_PACK` stays off unless you set it; the showdown card treats pack-on as not comparable.

```
cargo run -p myos_runtime -- --weights path/to/model.safetensors
cargo run -p myos_runtime -- --weights path/to/model.safetensors --steps 4
MYOS_WEIGHTS=path/to/model.safetensors cargo run -p myos_runtime
cargo run -p myos_runtime -- --weights path/to/model.safetensors --prompt-ids arena/prompts/s1_prompt.token_ids.txt --steps 128 --threads 8 --ref-ids runs/s1-llamacpp-bf16-b11960/token_ids.txt --fingerprint-dir runs/e2-myos-s1
cargo run -p myos_runtime --release -- --arena-s1 --weights path/to/model.safetensors --prompt-ids arena/prompts/s1_prompt.token_ids.txt --ref-ids runs/s1-llamacpp-bf16-b11960/token_ids.txt --threads 8
```

`--weights` wins over `MYOS_WEIGHTS` when both are set. `--block-fixture` still runs the in-process fixture when that flag is set. If a weights path is also set, the file is only checked as SafeTensors; its tensors are not the fixture. A bad path still exits before the fixture line.

`--fingerprint-dir DIR` writes `DIR/run_fingerprint.json` instead of `runs/<id>/`. It needs `--weights` or `--block-fixture`.

The fingerprint is host-only. It records the protocol §1.3 fields this process knows, plus `quant_or_dtype=bf16`, `acc_path=bf16_w_fp32_acc`, and `gemv_pack` (false unless `MYOS_GEMV_PACK=1`). `threads` is `--threads` (default 8). `model_sha256` and `seed` are JSON null (the file is not hashed, and greedy argmax has no RNG). `scenario_id` is `S1`. Without `--prompt-ids`, `n_prompt` is 1. With `--prompt-ids`, `n_prompt` is the file length, `n_gen` is `--steps`, and the notes say this is the E2 multi-token path. `prompt_token` is the first prompt id. `commit` is git HEAD when a `.git` directory is visible from the working directory, otherwise null. The fingerprint file is not itself a llama.cpp exact result; `--ref-ids` prints the CORRECT gate. `runs/` is gitignored. The showdown ledger is `arena/phase1-scoreboard.md` and `arena/s1-myos-bf16-cost-card.md`. On a machine without the 1.5B weights those MyOS medians stay UNMEASURED.

## SafeTensors loader

`cargo test -p weight_loader` covers `SafeTensorsWeights::parse` with a tiny in-process fixture (no model download). The `safetensors` crate is built with default features off, so its `std` feature stays disabled. Tensor bytes are borrowed from the caller's buffer. Mapped storage dtypes are `F32`, `F16`, and `BF16`.

The runtime smoke above is unchanged: it still loads `MmapWeights` via `cycle_fixture`, not a SafeTensors file.

## Qwen2.5 weight keys

`model_core::qwen25_layout` names every Qwen2.5-1.5B-Instruct tensor (`model.embed_tokens.weight`, `model.layers.{i}.*` for `i = 0..27`, and `model.norm.weight`) and the shapes from `docs/qwen25-1.5b-instruct-layout.md`. There is no `lm_head` key (`tie_word_embeddings`). `model_core::qwen25_bf16` borrows those payloads as little-endian BF16 byte views from any `WeightMap`, including `SafeTensorsWeights`, and rejects a missing name, a non-BF16 dtype, or a shape that is not the layout shape. It does not decode the bytes to `f32`. `QwenStub` still uses the f32 cycle fixture.

## BF16 numeric path

Phase1 main-track weights stay BF16 byte views. `tensor_core::gemv_bf16` is a row-major little-endian BF16 matrix times an f32 vector, accumulated in f32. `tensor_core::embed_gather_bf16` copies one `[vocab, hidden]` row into an f32 vector. Both decode BF16 (the top 16 bits of f32) inside the op. They do not call `load_f32_tensor`, and `load_f32_tensor` stays F32-only.

Fingerprint intent, hash later: `quant_or_dtype=bf16`, `acc_path=bf16_w_fp32_acc`.

`tensor_core::add_bf16_bias` adds one little-endian BF16 vector onto an f32 vector after a projection has finished (`y[i] +=` the promoted element, one lane at a time). Qwen2.5 Q/K/V biases use it. `o_proj` and the MLP have no bias.

`gemv_f32` and the `cycle_fixture` smoke stay on the f32 fixture. `cargo run -p myos_runtime` is still `[1, 2, 3, 0]`. Unit tests build tiny BF16 tables in process; they do not download Qwen.

## RMSNorm

`tensor_core::rmsnorm_bf16_weight` is the next main-track op. Activations stay f32 across the op. Gamma is a little-endian BF16 vector (`2 * n` bytes), promoted with `bf16_le_to_f32` inside the loop. `rmsnorm_f32` is the same kernel with an f32 gamma, for exact comparisons.

`Σx²` is one thread, left to right: `sum += x[i] * x[i]` (multiply, then add; not `mul_add`, not a tree). Then `inv_rms = 1 / sqrtf(sum / n + eps)` (IEEE-754 sqrt, not an approximate rsqrt) and `y[i] = w[i] * (x[i] * inv_rms)`. That reduction is the exact-sensitive part of `acc_path=bf16_w_fp32_acc` for this op: BF16 weights, f32 accumulation, no second dtype and no parallel sum. Qwen2.5 passes `RMS_NORM_EPS_QWEN25` (`1e-6`, `config.json` `rms_norm_eps`). Length mismatches return `Err` and leave `y` unchanged. The smoke line is still `[1, 2, 3, 0]`.

## RoPE

`tensor_core::rope_f32` and `rope_f32_inplace` rotate one head or a batch of heads that share one position. Activations stay f32. Qwen2.5 passes `ROPE_THETA_QWEN25` (`1e6`). Qwen2.5-1.5B passes `HEAD_DIM_QWEN25_1_5B` (`128` = 1536 / 12). `rope_inv_freqs` writes the pair table those rotates use.

Pairing is NeoX / half-rotate, the same layout as HF Qwen2 `rotate_half` and llama.cpp `LLAMA_ROPE_TYPE_NEOX`. With `h = head_dim / 2`, lane `i` rotates with lane `i + h`: `y[i] = x[i] cos - x[i+h] sin`, `y[i+h] = x[i+h] cos + x[i] sin` (multiply then add or subtract, no `mul_add`). The same angle is used on both halves. Adjacent even/odd lanes are not a second entry point.

`inv_freq[i] = 1 / powf(theta, (2i) / head_dim)`, then `angle = f32(position) * inv_freq[i]` and libm `cosf` / `sinf`. Each pair is independent (no iterative `freq *= theta**(-2/d)`). Position 0 is the identity on finite inputs. An odd `head_dim`, an empty buffer, or a length that is not a positive multiple of `head_dim` returns `Err` and does not write. The smoke line is still `[1, 2, 3, 0]`.

## GQA attention

`tensor_core::gqa_attn_f32` is one query position against cached K and V. Activations stay f32 (Q, K, V, scores, probabilities, output). There is no weight byte view: Q/K/V projections are `gemv_bf16` in `model_core::attention_layer`. Counts sit on `GqaShape`. Qwen2.5-1.5B passes `NUM_ATTENTION_HEADS_QWEN25_1_5B` (12), `NUM_KV_HEADS_QWEN25_1_5B` (2), and `HEAD_DIM_QWEN25_1_5B` (128), so `GQA_Q_PER_KV_QWEN25_1_5B` is 6. Query head `h` reads KV head `h / 6` (HF `repeat_kv`, contiguous groups).

Layout, no batch axis: Q and Y are `[num_q_heads, head_dim]`. K and V are `[num_kv_heads, seq_len, head_dim]`. `scale = 1 / sqrtf(head_dim)` once (`attn_scale_f32`), then each score is `dot * scale` after a left-to-right dot (multiply, then add; not `mul_add`). Softmax is one thread over the visible keys: max starts at key 0 (`>`), `expf(score - max)`, left-to-right sum, then `e / sum` (division, not a reciprocal multiply). The value mix is the same left-to-right order. `probs` is scratch of length `seq_len`; on success it holds the last query head.

`GqaMask::Full` scores the whole cache (decode that already stored only the legal prefix). `GqaMask::Causal { query_pos }` drops key index `t > query_pos` before the dot, so those V slots are not read. `query_pos >= seq_len - 1` keeps every key. A zero count, a query-head count that is not a multiple of the KV-head count, or a bad buffer length returns `Err` and does not write. A multi-query prefill kernel is not in this step. Per-layer KV allocation for that prefix is `model_core::KvCache` (below). The smoke line is still `[1, 2, 3, 0]`.

## SwiGLU

`tensor_core::swiglu_bf16` is one token's Qwen2 MLP. Activations stay f32. `mlp.gate_proj.weight`, `mlp.up_proj.weight`, and `mlp.down_proj.weight` stay little-endian BF16 views and go through `gemv_bf16` (row-major `[out, in]`, f32 dot, left to right, multiply then add). `swiglu_f32` is the same graph on f32 weights. Qwen2.5-1.5B passes `HIDDEN_SIZE_QWEN25_1_5B` (1536) and `INTERMEDIATE_SIZE_QWEN25_1_5B` (8960): gate and up are `[8960, 1536]`, down is `[1536, 8960]`. Counts sit on `SwigluShape` so a test can shrink them.

`gate = gemv(gate_proj, x)`, `up = gemv(up_proj, x)`, `h[i] = silu(gate[i]) * up[i]`, `y = gemv(down_proj, h)`. None of the three projections adds a bias (Qwen2.5 `mlp.*` has none). `silu_f32` is `z / (1 + expf(-z))` (libm `expf`, one division). That is llama.cpp `ggml_silu_f32`. It is not `z * (1 / (1 + expf(-z)))`; those differ by an ulp. The product is one multiply after SiLU returns, not `mul_add`. `gate`, `up`, and `h` are caller scratch of length `intermediate`. On success they hold those three vectors. A zero dimension or a bad buffer length returns `SwigluError::Shape` and does not write. `forward_token` calls this op after `post_attention_layernorm` and adds the result onto the residual (knife ③, below). The smoke line is still `[1, 2, 3, 0]`.

## KV cache and attention wiring

Knife ② is one attention layer. `model_core::KvCache` stores K and V for every layer. Qwen2.5-1.5B is 28 layers, 2 KV heads, head dim 128 (`KvShape::qwen25_1_5b`). Counts sit on `KvShape` so a test can shrink them. Every layer `store`s the current token at the same cursor. The caller `advance`s once after that token. `attention_layer` stores and does not advance. That cursor is ember's `KvCache` (MIT, Copyright (c) 2026 Ali Zulfiqar, `src/attention.rs`, allocated from `Model::new_cache` in `src/model.rs`).

Each slot is `[kv_head, pos, head_dim]` with a `max_seq` stride. The layer packs the committed prefix plus the token just projected into `[kv_head, seq, head_dim]` and calls `gqa_attn_f32` with `GqaMask::Full`. Scores stay in that op (`e / sum`).

`attention_layer` borrows `q/k/v/o` weights and the `q/k/v` biases through `qwen25_bf16` (little-endian BF16 views). Shapes come from `AttnShape`, so a fixture can be smaller than 1536. `gemv_bf16` writes each projection, then `add_bf16_bias` adds the Q/K/V bias in f32. `o_proj` has no bias. Q and K go through `rope_f32_inplace` (NeoX, one position, every head). V is not rotated. The output GEMV is added onto `residual`. `proj_in` is the QKV input. This step does not RMSNorm and does not run SwiGLU. Passing the same values for `proj_in` and the residual skips the norm. The full block is knife ③.

## Transformer block (knife ③)

`model_core::forward_token` is one token through every layer, then the tied head. The loop follows ember `Model::forward` (MIT, Copyright (c) 2026 Ali Zulfiqar, `src/model.rs`): `input_layernorm`, `attention_layer` (which adds `o_proj` onto the residual), `post_attention_layernorm`, `swiglu_bf16` (`down(SiLU(gate) ⊙ up)`, no MLP bias), residual add. After the last layer, `model.norm.weight` and `gemv_bf16` on `model.embed_tokens.weight`. There is no `lm_head` key. Weights stay little-endian BF16 views (`acc_path=bf16_w_fp32_acc`). Counts sit on `BlockShape`, so a test can shrink them. Qwen2.5-1.5B passes `BlockShape::qwen25_1_5b` (28 layers, hidden 1536, intermediate 8960, vocab 151936, `rms_norm_eps` 1e-6).

`attention_layer` still does not advance the cache. `forward_token` stores every layer at the current cursor, then `KvCache::advance` once. `prefill` feeds prompt token ids. `greedy_decode` feeds `greedy_token` back in and does not advance a second time. Phase1 core takes token ids; it does not tokenize. Prefill here is one token at a time, not a multi-query kernel.

`QwenStub` still counts tokens on `StubCursor`. `cargo run -p myos_runtime` is still `[1, 2, 3, 0]`. `--block-fixture` runs `block_fixture` (hidden 2, vocab 4, one layer, built in process) and prints `[1, 1, 1, 1]`. Unit tests build tiny BF16 tables in process; they do not download Qwen.

## Weights file (knife ④)

`cargo run -p myos_runtime -- --weights PATH` maps that file with `SafeTensorsMmap` and runs the same `prefill` / `greedy_decode` at `BlockShape::qwen25_1_5b`. The bytes stay a BF16 view (`quant_or_dtype=bf16`, `acc_path=bf16_w_fp32_acc`). The host does not call `load_f32_tensor` on those tensors and does not build a `Vec<f32>` of the checkpoint. See the smoke section for the CLI line and `runs/<id>/run_fingerprint.json`. A few-hundred-byte SafeTensors copy of `block_fixture` is what CI decodes twice, at `fixture_shape`, to check that the mmap path is stable. The CLI itself still demands the 1.5B shape, so that miniature file is an error, not a successful weights line.

## Feature flags

`weight_loader` is `no_std` + `alloc` by default. `tensor_core` is `no_std` unless the `std` feature is on. `myos_runtime` turns that feature on for the GEMV row pool only. `model_core` stays `no_std` and does not enable host I/O.

| Feature | When | What it adds |
| --- | --- | --- |
| (default) | `no_std` library builds | `SafeTensorsWeights::parse(&[u8])` over a caller-owned buffer. No OS, no mmap. |
| `host` | non-test builds that open files (`myos_runtime` enables it) | `std` and `memmap2`. `SafeTensorsMmap::open(path)` maps a `.safetensors` file read-only. `parse()` borrows tensor payloads from the map; it does not copy them into a `Vec<u8>`. |

`myos_runtime` enables `weight_loader`'s `host` feature so `--weights` can map a SafeTensors file. `parse()` borrows tensor payloads from the map. Do not turn `host` on for a `no_std` target: it links `std`. `cargo test -p weight_loader` still compiles the mmap module for that crate's own tests; `--features host` is what dependents need outside tests. The fingerprint writer lives in the runtime binary, not in `model_core`.

Tests generate a few-hundred-byte fixture at runtime (same `safetensors::serialize` path as the in-memory parser tests). Real `*.safetensors` weights stay gitignored.

## Team

GitHub 写操作仅由 OSdeveloper 负责；博弈手 / 仓寻 / 深析 / 联调官负责方案、搜索、分析与联调结论。
