# Phase1 Scoreboard（评测账本）

**协议**：`PHASE1-ARENA-PROTOCOL-v0.1.md`  
**更新**：2026-11-03 UTC+8 · 博弈手  
**合入**：交 OSdeveloper 推仓（本文件不直接 git push）

---

## 总览

| 里程碑 | 状态 | 说明 |
|--------|------|------|
| E0 冻结 | ✅ | v0.1：1.5B / exact / S3=4096 / CPU / ±5% |
| E1 llama 基线 S1 | 🟡 部分 | 主赛道 BF16+FP16 DET-PASS；副赛道 Q4 DET-FAIL 仅参考 |
| E2 MyOS exact S1 | ✅ | CORRECT PASS vs BF16 金标准 |
| 主赛道 S1 吞吐对打 | ⬜ 未开 | E2 通过后可派 |
| 副赛道 Q4 对打 | ⬜ 未开 | 协议：不对打直至另议 |

---

## 主赛道 · S1（BF16 优先金标准）

| 实现 | run_id | DET | CORRECT | T_decode | L_TTFT ms | L_TPOT ms | M_peak MiB | 备注 |
|------|--------|-----|---------|----------|-----------|-----------|------------|------|
| llama.cpp b11960 | `s1-llamacpp-bf16-b11960` | PASS | 金标准 | 17.23 | 608.1 | 58.02 | 3051 | 自转 GGUF BF16；E2 ref |
| MyOS `42acbb39` | `e2-myos-s1` | （单次 greedy） | **PASS** exact 128/128 | — | — | — | — | 仅正确性收口；无正式 R=5 cost card |
| llama.cpp b11960 | `s1-llamacpp-fp16-b11960` | PASS | （非 E2 ref） | 18.84 | 841.1 | 53.07 | 6377 | 官方 FP16；并列主赛道参考，非本 E2 金标准 |

胜负列：吞吐对打未开 → 无 W/T/L。

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

1. 派主赛道 S1 **正式 cost-card 对打**（MyOS 补 metrics：W/R、T_decode/L_*/M_peak，threads 对齐规则待实现侧说明）
2. E1 补 S2/S3/S6 基线（若尚未闭合）
3. 副赛道保持关闭，除非月星/协议升版

