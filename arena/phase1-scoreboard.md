# Phase1 Scoreboard（评测账本）

**协议**：`PHASE1-ARENA-PROTOCOL-v0.1.md`  
**更新**：2026-11-03 · 刀⑦已合 main @ `a92677f1`；刀⑧ Prefill/TTFT 代码落地，中位 UNMEASURED  
**合入**：刀⑦ PR #39 已在 main。刀⑧ 指纹另开，host 补测前不改刀⑦数字。

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
| MyOS threads=8（刀⑦） | `s1-myos-bf16-t8` | **PASS** | **PASS** | 14.3503 | 9092.903 | 69.685 | 2963.383 | host 2026-11-03 15:02 UTC+8；`bf16` / `bf16_w_fp32_acc`，batch=1，`gemv_pack=false`；commit `9577e66`。卡：`arena/s1-myos-bf16-cost-card.md`。已合 main @ `a92677f1` |
| MyOS threads=8（刀⑧ prefill） | `s1-myos-bf16-t8-k8` | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | 整段 chunk prefill 已落地；exact 由单测对 serial `forward_token`。本环境无 1.5B 权重。卡：`arena/s1-myos-bf16-t8-k8-cost-card.md`。口径：`docs/phase1-knife7-verdict-and-knife8.md` |
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

1. 刀⑦已合 main @ `a92677f1`（PR #39）。账本刀⑦行与 `arena/s1-myos-bf16-cost-card.md` 是该收口的 host 中位。
2. 刀⑧ Prefill/TTFT：整段 prompt 分块 prefill，指纹 `s1-myos-bf16-t8-k8`。本环境 CORRECT/DET/中位 **UNMEASURED**。host `--arena-s1` 补测命令见 `docs/phase1-knife7-verdict-and-knife8.md`。
3. E1 补 S2/S3/S6 基线（若尚未闭合）
4. 副赛道保持关闭，除非月星/协议升版
