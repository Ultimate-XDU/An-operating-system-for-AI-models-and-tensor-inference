# 刀⑫口径（2026-11-04）

main 上没有刀⑪判决稿。本文件只钉刀⑫，不补写刀⑪实测，也不把刀⑩⑪代码算进这条基线。基线是已合入的刀⑨：`T_decode` **15.4880**，指纹 `s1-myos-bf16-t8-k9`。

## 刀⑫改了什么

一口粗改，默认路径没有新开关。

- `KvCache` 仍是 `[num_kv_heads, max_seq, head_dim]`。decode 不再每步 `pack_prefix_with_token`。已提交的前缀按 `kv_stride = max_seq` 直接读；当前 token 的 K/V 放在旁边，注意力接受之后才 `store`。
- prefill 仍是刀⑧的整段 chunk。K/V 已经 `store_at` 进 cache，`attend_chunk` 用同一 stride 读，不再 `pack_prefix`。因果掩码仍是前缀。
- `gqa_attn_f32` 的点积、softmax、每条输出 lane 的累加顺序不变：先乘后加，不 `mul_add`，softmax 仍是 max、`expf(score - max)`、从左到右求和、`e / sum`。同一 KV head 的 query（1.5B 为 6）一起走这些链；V 的多条 lane 一起走，每条 lane 仍按 token 从左到右加。
- 不改 GEMV 行宽 / 寄存器块，不做 fused multi-GEMV，不开 pack / 预取开关，不改乘加序，不调 threads，不开 S2 / Q4。

## 硬门（host；本环境 UNMEASURED）

| 硬门 | 门槛 |
|------|------|
| CORRECT / DET | PASS @ threads=8 |
| T_decode | ≥ **16.42**，或（≥ **16.37** 且场景加权严格优于 −80） |
| L_TTFT | ≤ **2193.6** |
| 相对刀⑨ | T_decode ≥ **15.4880** |

指纹 `runs/s1-myos-bf16-t8-k12/`。草稿卡 `arena/s1-myos-bf16-t8-k12-cost-card.md` 在本环境是 UNMEASURED（没有 1.5B 权重）。host 用下面的命令覆盖卡片和指纹；覆盖前不要把卡片上的空位读成平带或场景胜。

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
  --fingerprint-dir runs/s1-myos-bf16-t8-k12 \
  --cost-card arena/s1-myos-bf16-t8-k12-cost-card.md
```
