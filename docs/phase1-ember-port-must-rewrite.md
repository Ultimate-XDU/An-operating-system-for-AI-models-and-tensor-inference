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
