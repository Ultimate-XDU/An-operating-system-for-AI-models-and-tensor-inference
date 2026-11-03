# Phase1 Scoreboard（评测账本）

**协议**：`PHASE1-ARENA-PROTOCOL-v0.1.md`  
**更新**：2026-11-03 UTC+8 · 刀⑦ S1 对打槽（threads=8；MyOS 中位待 host）  
**合入**：交 OSdeveloper 推仓（本文件不直接 git push）

---

## 总览

| 里程碑 | 状态 | 说明 |
|--------|------|------|
| E0 冻结 | ✅ | v0.1：1.5B / exact / S3=4096 / CPU / ±5% |
| E1 llama 基线 S1 | 🟡 部分 | 主赛道 BF16+FP16 DET-PASS；副赛道 Q4 DET-FAIL 仅参考 |
| E2 MyOS exact S1 | ✅ | CORRECT PASS vs BF16 金标准（当时 threads=1） |
| 主赛道 S1 吞吐对打 | 🟡 闭环已接线，中位未测 | 刀⑦：threads=8 行维并行 + W=3/R=5 harness。本环境无 1.5B 权重，MyOS 中位 UNMEASURED，场景按 §4.3 **不可比**（不是空行，也不是 W/T/L） |
| 副赛道 Q4 对打 | ⬜ 未开 | 协议：不对打直至另议 |

---

## 主赛道 · S1（BF16 优先金标准）

| 实现 | run_id | DET | CORRECT | T_decode | L_TTFT ms | L_TPOT ms | M_peak MiB | 备注 |
|------|--------|-----|---------|----------|-----------|-----------|------------|------|
| llama.cpp b11960 | `s1-llamacpp-bf16-b11960` | PASS | 金标准 | 17.2348 | 608.110 | 58.022 | 3050.953 | 自转 GGUF BF16；`-t 8`；E2 ref；ratio 分母 |
| MyOS threads=8（刀⑦） | `s1-myos-bf16-t8` | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | 正式对打槽。`bf16` / `bf16_w_fp32_acc`，batch=1，`gemv_pack=false`。与上一行同 `n_threads=8`。卡：`arena/s1-myos-bf16-cost-card.md` |
| MyOS `42acbb39` | `e2-myos-s1` | （单次 greedy） | **PASS** exact 128/128 | — | — | — | — | 历史正确性收口；指纹 threads=1，**不**拿来和 `-t 8` 比吞吐 |
| llama.cpp b11960 | `s1-llamacpp-fp16-b11960` | PASS | （非 E2 ref） | 18.84 | 841.1 | 53.07 | 6377 | 官方 FP16；并列主赛道参考，非本 E2 金标准 |

胜负列（S1 BF16，MyOS threads=8 vs llama `-t 8`）：

| 指标 | MyOS | llama | ratio (MyOS/base) | 判定 |
|------|------|-------|-------------------|------|
| T_decode | UNMEASURED | 17.2348 | — | UNMEASURED |
| L_TTFT | UNMEASURED | 608.110 | — | UNMEASURED |
| L_TPOT | UNMEASURED | 58.022 | — | UNMEASURED |
| M_peak | UNMEASURED | 3050.953 | — | UNMEASURED |
| 场景 | — | — | — | **不可比**（§4.3：挑战者中位未跑出）。权重仍是 T_decode 0.35 / L_TPOT 0.25 / L_TTFT 0.20 / M_peak 0.20。host 填上 MyOS 中位后：≥1.05 或 ≤0.95 才是 W 或 L，其余 T；CORRECT 或 DET 失败则场景负且吞吐作废。 |

本环境没有宣称 ±5% 胜。软记录（GEMV 微基准，不是 S1）：gate/up `[8960, 1536]`，threads=1 → 8 为 3.611×（4 cores）。见 `docs/phase1-knife7-closeout.md`。

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

1. host 重跑刀⑦ `--arena-s1 --threads 8`（W=3 R=5），把 MyOS 中位填进 `arena/s1-myos-bf16-cost-card.md` 与上表。未测前场景保持不可比。命令见 `docs/phase1-knife7-closeout.md`。
2. E1 补 S2/S3/S6 基线（若尚未闭合）
3. 副赛道保持关闭，除非月星/协议升版

