# Qwen2.5-1.5B-Instruct layout pin (Phase1)

| 项 | 内容 |
|----|------|
| 文档 | `docs/qwen25-1.5b-instruct-layout.md` |
| 状态 | Phase1 主赛道权重布局钉死 |
| 来源 | 仓寻核对的 Hub 元数据；本文件按同一 revision 复核 |
| 日期 | 2026-11-02 |
| 效力 | 布局事实。与对打协议冲突时以 `PHASE1-ARENA-PROTOCOL-v0.1.md` 为准 |
| 代码 | `model_core::qwen25_layout` 只放常量。`QwenStub` / `cycle_fixture` 仍用玩具键，不读这些常量 |

复核方式（未下载 3GB 权重）：Hub `tree/main` 给出文件大小与 LFS oid；对 `model.safetensors` 只做 HTTP Range，读 8 字节头长加 38,528 字节 JSON header。模型仓 revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`。`config.json` 与模型卡 README 同一次读取。

---

## Source

| 项 | 值 |
|----|----|
| Model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Card | https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct |
| Weights file | **single** `model.safetensors`（未分片）。仓库内没有 `model.safetensors.index.json`，也没有 `model-0000N-of-M.safetensors` |
| Size | **3,087,467,144** B ≈ **2.88 GiB**（3,087,467,144 / 1024³ ≈ 2.876 GiB） |
| LFS SHA256 | `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee` |
| Git blob oid | `9127f71e7314df0064f469223749e5f237e06463`（LFS pointer，不是内容哈希。对打指纹用上面的 LFS SHA256） |
| Resolve URL | https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/resolve/main/model.safetensors |
| Official `torch_dtype` | **bfloat16**（`config.json`）。文件内 **338/338** 个张量 dtype 均为 SafeTensors **`BF16`** |
| Header | 8-byte LE length + 38,528-byte JSON。`__metadata__.format = "pt"`。张量 payload = 1,543,714,304 × 2 = 3,087,428,608 B。8 + 38,528 + payload = 文件大小 |
| Phase1 main track | **BF16**。FP16 只作为可选转换，**不是**本文件的默认存储，也不要当成主赛道默认档 |
| Param count (BF16) | **1,543,714,304**（tied embedding 只计一次） |
| Tensor count | **338** |
| `tie_word_embeddings` | `true` → **没有**独立的 `lm_head.*`。输出投影用 `model.embed_tokens.weight` |

参数拆分（与 header 元素个数一致）：

| 块 | 计算 | 参数 |
|----|------|------|
| `model.embed_tokens.weight` | 151936 × 1536 | 233,373,696 |
| `model.norm.weight` | 1536 | 1,536 |
| 每层（见下表，含 Q/K/V bias，不含 o_proj bias） | | 46,797,824 |
| 28 层 | 46,797,824 × 28 | 1,310,339,072 |
| 合计 | | **1,543,714,304** |

---

## Config highlights

来自 `config.json`（`Qwen2ForCausalLM` / `model_type=qwen2`）：

| 字段 | 值 |
|------|----|
| `hidden_size` | **1536** |
| `intermediate_size` | **8960** |
| `num_hidden_layers` | **28**（层下标 `i = 0..27`） |
| `num_attention_heads` | **12** |
| `num_key_value_heads` | **2**（GQA） |
| head dim | 1536 / 12 = **128**（配置里没有单独字段；由上两项推出） |
| K/V 输出维 | 2 × 128 = **256** |
| `rope_theta` | **1e6**（`1000000.0`） |
| `vocab_size` | **151936** |
| `tie_word_embeddings` | `true` |
| `torch_dtype` | `bfloat16` |
| `hidden_act` | `silu` |
| `rms_norm_eps` | `1e-6` |
| `max_position_embeddings` | 32768 |
| `sliding_window` / `use_sliding_window` | 32768 / `false` |
| `bos_token_id` / `eos_token_id` | 151643 / 151645 |
| `attention_dropout` | 0.0 |

模型卡同时写：参数量 1.54B，非 embedding 参数 1.31B，28 层，GQA 为 Q=12、KV=2，架构含 RoPE、SwiGLU、RMSNorm、Attention QKV bias、tied word embeddings。非 embedding 1.31B 与上表 28 层合计 1,310,339,072 一致。

---

## Tensor names

全局：

| 名称 | dtype | shape |
|------|-------|-------|
| `model.embed_tokens.weight` | BF16 | `[151936, 1536]` |
| `model.norm.weight` | BF16 | `[1536]` |

每一层 `model.layers.{i}.*`（`i = 0..27`，形状在 28 层上相同）：

| 名称 | dtype | shape |
|------|-------|-------|
| `input_layernorm.weight` | BF16 | `[1536]` |
| `self_attn.q_proj.weight` | BF16 | `[1536, 1536]` |
| `self_attn.q_proj.bias` | BF16 | `[1536]` |
| `self_attn.k_proj.weight` | BF16 | `[256, 1536]` |
| `self_attn.k_proj.bias` | BF16 | `[256]` |
| `self_attn.v_proj.weight` | BF16 | `[256, 1536]` |
| `self_attn.v_proj.bias` | BF16 | `[256]` |
| `self_attn.o_proj.weight` | BF16 | `[1536, 1536]` |
| `post_attention_layernorm.weight` | BF16 | `[1536]` |
| `mlp.gate_proj.weight` | BF16 | `[8960, 1536]` |
| `mlp.up_proj.weight` | BF16 | `[8960, 1536]` |
| `mlp.down_proj.weight` | BF16 | `[1536, 8960]` |

`self_attn.o_proj` **没有** bias。`mlp.*` **没有** bias。header 里不存在任何 `lm_head` 键。

每层 12 个张量 × 28 + 全局 2 = **338**。

SafeTensors 的二维布局是 `[out, in]`（PyTorch linear weight）。加载时不要把 K/V 的 256 当成 `num_key_value_heads` 本身：256 = 2 heads × 128 head dim。

### Tokenizer sidecars

推理接入 tokenizer 时还要这些仓库文件（本次不加载）：

- `tokenizer.json`
- `tokenizer_config.json`
- `vocab.json`
- `merges.txt`
- `config.json`

同目录还有 `generation_config.json` 与 `LICENSE`，不属于本布局表的张量键。

---

## Links

- Model card: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
- Base model: https://huggingface.co/Qwen/Qwen2.5-1.5B
- Qwen docs: https://qwen.readthedocs.io/en/latest/
- Blog: https://qwenlm.github.io/blog/qwen2.5/
- GitHub Qwen2.5: https://github.com/QwenLM/Qwen2.5
- Apache-2.0: https://www.apache.org/licenses/LICENSE-2.0（模型卡 `license: apache-2.0`；仓库内文本 https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/LICENSE ）
- arXiv: https://arxiv.org/abs/2407.10671 （Qwen2 Technical Report）

---

## Code pin

`crates/model_core/src/qwen25_layout.rs` 只导出层数、hidden、vocab，以及 embed / final-norm 两个键名。`QwenStub::forward_token` 与 `cycle_fixture` 不读取它们。烟雾测试仍是 `cargo test --workspace` 里的 cycle GEMV，不打开本权重文件。
