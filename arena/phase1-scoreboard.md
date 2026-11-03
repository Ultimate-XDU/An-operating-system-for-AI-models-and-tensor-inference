# Phase1 Scoreboard（评测账本）

**协议**：`PHASE1-ARENA-PROTOCOL-v0.1.md`  
**更新**：2026-11-03 17:40 UTC+8 · 刀⑦ host `--arena-s1` 已填中位（CORRECT/DET PASS，场景 L）  
**合入**：交 OSdeveloper 推仓（本文件不直接 git push）

---

## 总览

| 里程碑 | 状态 | 说明 |
|--------|------|------|
| E0 冻结 | ✅ | v0.1：1.5B / exact / S3=4096 / CPU / ±5% |
| E1 llama 基线 S1 | 🟡 部分 | 主赛道 BF16+FP16 DET-PASS；副赛道 Q4 DET-FAIL 仅参考 |
| E2 MyOS exact S1 | ✅ | CORRECT PASS vs BF16 金标准（当时 threads=1） |
| 主赛道 S1 吞吐对打 | ✅ 可比已落地 | 刀⑦ host W=3/R=5 中位已填。CORRECT/DET PASS；场景 **L**（加权和 -80）。卡：`arena/s1-myos-bf16-cost-card.md`；产物：`runs/s1-myos-bf16-t8/` |
| 副赛道 Q4 对打 | ⬜ 未开 | 协议：不对打直至另议 |

---

## 主赛道 · S1（BF16 优先金标准）

| 实现 | run_id | DET | CORRECT | T_decode | L_TTFT ms | L_TPOT ms | M_peak MiB | 备注 |
|------|--------|-----|---------|----------|-----------|-----------|------------|------|
| llama.cpp b11960 | `s1-llamacpp-bf16-b11960` | PASS | 金标准 | 17.2348 | 608.110 | 58.022 | 3050.953 | 自转 GGUF BF16；`-t 8`；E2 ref；ratio 分母 |
| MyOS threads=8（刀⑦） | `s1-myos-bf16-t8` | **PASS** | **PASS** | 14.3503 | 9092.903 | 69.685 | 2963.383 | host 2026-11-03 15:02 UTC+8；`bf16` / `bf16_w_fp32_acc`，batch=1，`gemv_pack=false`；commit `9577e66`。卡：`arena/s1-myos-bf16-cost-card.md` |
| MyOS `42acbb39` | `e2-myos-s1` | （单次 greedy） | **PASS** exact 128/128 | — | — | — | — | 历史正确性收口；指纹 threads=1，**不**拿来和 `-t 8` 比吞吐 |
| llama.cpp b11960 | `s1-llamacpp-fp16-b11960` | PASS | （非 E2 ref） | 18.84 | 841.1 | 53.07 | 6377 | 官方 FP16；并列主赛道参考，非本 E2 金标准 |

胜负列（S1 BF16，MyOS threads=8 vs llama `-t 8`）：

| 指标 | MyOS | llama | ratio (MyOS/base) | 判定 |
|------|------|-------|-------------------|------|
| T_decode | 14.3503 | 17.2348 | 0.8326 | L |
| L_TTFT | 9092.903 | 608.110 | 14.9527 | L |
| L_TPOT | 69.685 | 58.022 | 1.2010 | L |
| M_peak | 2963.383 | 3050.953 | 0.9713 | T |
| 场景 | — | — | 加权和 -80 | **L**（CORRECT/DET 均 PASS，吞吐可比；未达 ±5% 胜） |

权重：T_decode 0.35 / L_TPOT 0.25 / L_TTFT 0.20 / M_peak 0.20。这不是「已经胜过 llama」的对外句。软记录（GEMV 微基准，不是 S1）：gate/up `[8960, 1536]`，threads=1 → 8 为 3.611×（云端 4 cores）。见 `docs/phase1-knife7-closeout.md`。

---

## 副赛道 · S1 Q4_K_M（不对打）

| 实现 | run_id | DET | 用途 |
|------|--------|-----|------|
| llama.cpp -t8 | `s1-llamacpp-q4km-b11960` | FAIL | 吞吐参考 · 胜负/CORRECT 作废 |
| llama.cpp noAMX -t1 | `s1-llamacpp-q4km-b11960-det` | PASS | 附录指纹；未开对打 |

详见既有口径：DET-FAIL 不入主胜负。

---

## E2 条目指针

- 详单：`arena/e2-s1-correct.md`
- MyOS tokens：`runs/e2-myos-s1/token_ids.txt`
- Ref tokens：`runs/s1-llamacpp-bf16-b11960/token_ids.txt`

---

## 下一步（协议 §8）

1. ~~host 重跑刀⑦ `--arena-s1`~~ **已完成**（2026-11-03 15:02 UTC+8）。账本与 cost-card 已填。交 OSdeveloper 把 `arena/phase1-scoreboard.md` + cost-card 推上 [PR #39](https://github.com/Ultimate-XDU/An-operating-system-for-AI-models-and-tensor-inference/pull/39) 后再合入。
2. E1 补 S2/S3/S6 基线（若尚未闭合）
3. 副赛道保持关闭，除非月星/协议升版
4. 场景 L 的主因是 L_TTFT（约 15× llama）；下一刀若继续吞吐，应先钉 prefill 路径，而不是再拆标量 GEMV 微刀
