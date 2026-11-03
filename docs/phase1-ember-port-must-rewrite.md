# Phase1：从 ember 移植时必须改写（刀①硬红线）

| 项 | 内容 |
|----|------|
| 文档 | `docs/phase1-ember-port-must-rewrite.md` |
| 状态 | 已拍板。刀①及后续移植服从本文 |
| 参考 | ember `src/model.rs` `Model::load` 只作控制流与键名参考：https://github.com/codewithfourtix/ember （MIT, Copyright (c) 2026 Ali Zulfiqar） |
| 效力 | 与 `PHASE1-ARENA-PROTOCOL-v0.1.md` 冲突时以协议为准 |
| 不做 | 不把 ember 整仓 fork 进 MyOS。不抄 `load_f32`、rayon、tokenizers |

ember 的 `load_f32` 在加载期把 BF16/F16/F32 整表收成 `Vec<f32>`。MyOS 主路不走那条。下面三条是原文，实现必须服从。

## 1. BF16

> BF16：权重保持 BF16 字节视图（mmap/caller buffer）；升精只在算子内→f32 累加；指纹口径 quant_or_dtype=bf16 + acc_path=bf16_w_fp32_acc。禁止加载期整表 Vec<f32>，禁止在映射胶水里 decode 偷换主路。

刀①落地：`weight_loader::bf16_view` 与 `model_core::qwen25_bf16` 只归还 little-endian BF16 `&[u8]` 加形状。不调用 `load_f32_tensor`，不在胶水里做 BF16→f32。`load_f32_tensor` 仍只接受 F32，供 cycle fixture 使用。

## 2. exact

> exact（刀①主要是键名+视图，但别埋下违例）：不要引入 rayon；不要另起数值方言。

刀①不新增归约、不并行、不换一套 BF16 解码。升精仍只在已有 `tensor_core` 算子内（`acc_path=bf16_w_fp32_acc`）。

## 3. no_std

> no_std：控制流/映射进 model_core；mmap/CLI/tokenizers 不进 core；勿把 anyhow/rayon/tokenizers/onig 加进 core。

键名在 `model_core::qwen25_layout`，借用在 `model_core::qwen25_bf16`。`model_core` 库依赖仍只有 `tensor_core` 与 `weight_loader`（默认无 `host`）。mmap 留在 `weight_loader` 的 `host` feature，由 `myos_runtime` 打开。单测里的微型 SafeTensors 夹具是 `model_core` 的 dev-dependency，不进入库依赖。

## 刀② 状态

刀②已落地（KV cache + 一层 attention 接线），仍服从上面三条。

- `model_core::KvCache`：每层 K/V。Qwen2.5-1.5B 为 28 层、2 个 KV 头、head dim 128。形状在 `KvShape` 上，单测可缩小。同一 token 的每一层都写在当前 cursor，调用方在该 token 结束后 `advance` 一次。`attention_layer` 自己不 `advance`。控制流参考 ember `src/attention.rs` `KvCache` 与 `src/model.rs` `Model::new_cache`（MIT, Copyright (c) 2026 Ali Zulfiqar）。槽布局是 `[kv_head, pos, head_dim]`，打成 `[kv_head, seq, head_dim]` 后交给已有的 `gqa_attn_f32`（`e / sum`）。
- `model_core::attention_layer`：经 `qwen25_bf16` 借 q/k/v/o 权重和 q/k/v bias（BF16 字节，形状可缩小）。`gemv_bf16`，再 `tensor_core::add_bf16_bias`（升精在算子内），Q/K 走 `rope_f32_inplace`（NeoX），V 不旋转，写入 cache，`gqa_attn_f32`，`o_proj` 无 bias，残差相加。不做 RMSNorm，不做 SwiGLU，不循环 28 层。
- 整块 forward、tied `lm_head`、prefill/decode 已由刀③接上。

## 刀③ 状态

刀③已落地，仍服从上面三条。

- `model_core::forward_token`：每一层 `input_layernorm` → `attention_layer`（`o_proj` 残差在该函数里加）→ `post_attention_layernorm` → `swiglu_bf16` → 残差。全部层之后是 `model.norm.weight`，再用 `gemv_bf16` 乘 `model.embed_tokens.weight`。没有 `lm_head` 键。升精只在已有算子内。控制流参考 ember `Model::forward`（MIT, Copyright (c) 2026 Ali Zulfiqar，`src/model.rs` 的 block 循环、final norm、tied lm_head）和 `src/main.rs` 的 prefill / greedy，但只吃 token id。不抄 `load_f32`、rayon、tokenizers、`*(1/sum)` softmax、interleaved RoPE。
- `attention_layer` 仍不 `advance`。`forward_token` 在该 token 每一层都 `store` 之后调用一次 `KvCache::advance`。`prefill` / `greedy_decode` 不再次 advance。
- `myos_runtime` 默认烟雾仍是 `[1, 2, 3, 0]`。`--block-fixture` 另走一条进程内微型 BF16 夹具上的 greedy，不下载 1.5B。

`QwenStub` 仍用 `StubCursor` 计数。烟雾行仍是 `[1, 2, 3, 0]`。

## 刀④ 状态

刀④已落地（真 SafeTensors 接到已有 block forward），仍服从上面三条。不新写算子。

- `myos --weights PATH`（没有 `--block-fixture`）用已有 `SafeTensorsMmap` 映射文件，形状是 `BlockShape::qwen25_1_5b()`，然后走已有 `prefill` / `greedy_decode`。权重仍是 BF16 字节视图。不调用 `load_f32_tensor`，不在 runtime 里把整表收成 `Vec<f32>`。升精只在已有 `tensor_core` 算子内（NeoX、QKV bias、SwiGLU、串行核都不换）。
- 默认无参数仍是 `[1, 2, 3, 0]`，不写指纹。`--block-fixture` 仍是进程内微型夹具 `[1, 1, 1, 1]`。两条路径都不下载 1.5B。`--weights` 与 `--block-fixture` 同时给出时，解码仍是夹具；文件只做 SafeTensors 解析，坏路径在夹具行之前退出。
- 宿主写出 `runs/<id>/run_fingerprint.json`（权重跑 `s1-weights`，夹具跑 `s1-block-fixture`）。`quant_or_dtype=bf16`，`acc_path=bf16_w_fp32_acc`，`deterministic=true`，`device=cpu`，`threads=1`。`model_sha256` 与 `seed` 为 JSON null（不把布局文档里的 LFS oid 抄进指纹，也不假装有 RNG seed）。`scenario_id` 是 `S1`。未给 `--prompt-ids` 时 `n_prompt` 是 1；给出时 `n_prompt` 是文件长度、`n_gen` 是 `--steps`（协议 S1 为 128/128），并写 `token_ids.txt`。`--ref-ids` 打印 `CORRECT`，不匹配仍退出 0。指纹文件本身不是 llama.cpp exact 结果。
- mmap、CLI、指纹留在 `myos_runtime`。`model_core` 仍是 no_std，依赖不变。没有 rayon、anyhow、tokenizers、onig。
