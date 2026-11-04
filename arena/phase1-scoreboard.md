# Phase1 Scoreboard（评测账本）

**协议**：`PHASE1-ARENA-PROTOCOL-v0.1.md`  
**更新**：2026-11-04 · 刀⑨ host 中位已填（指纹 `s1-myos-bf16-t8-k9`，W=3/R=5）；硬门未过。刀⑪草稿 UNMEASURED（指纹 `s1-myos-bf16-t8-k11`），对照仍用刀⑨ T_decode **15.4880**。  
**合入**：刀⑧已在 main @ `569b55e1`（PR #41）。刀⑨不改刀⑦/刀⑧已填中位。刀⑪不改刀⑨已填中位。

---

## 总览

| 里程碑 | 状态 | 说明 |
|--------|------|------|
| E0 冻结 | ✅ | v0.1：1.5B / exact / S3=4096 / CPU / ±5% |
| E1 llama 基线 S1 | 🟡 部分 | 主赛道 BF16+FP16 DET-PASS；副赛道 Q4 DET-FAIL 仅参考 |
| E2 MyOS exact S1 | ✅ | CORRECT PASS vs BF16 金标准（当时 threads=1） |
| 主赛道 S1 吞吐对打 | ✅ 可比已落地 | 刀⑦与刀⑧ host W=3/R=5 中位均已填。刀⑧ CORRECT/DET PASS；L_TTFT 2193.653；场景仍 **L**（加权和 -80）。刀⑨ host 已填：T_decode 15.4880、场景仍 −80，硬门未过。刀⑪草稿 UNMEASURED。卡：刀⑦ `arena/s1-myos-bf16-cost-card.md`；刀⑧ `arena/s1-myos-bf16-t8-k8-cost-card.md`；刀⑨ `arena/s1-myos-bf16-t8-k9-cost-card.md`；刀⑪ `arena/s1-myos-bf16-t8-k11-cost-card.md` |
| 副赛道 Q4 对打 | ⬜ 未开 | 协议：不对打直至另议 |

---

## 主赛道 · S1（BF16 优先金标准）

| 实现 | run_id | DET | CORRECT | T_decode | L_TTFT ms | L_TPOT ms | M_peak MiB | 备注 |
|------|--------|-----|---------|----------|-----------|-----------|------------|------|
| llama.cpp b11960 | `s1-llamacpp-bf16-b11960` | PASS | 金标准 | 17.2348 | 608.110 | 58.022 | 3050.953 | 自转 GGUF BF16；`-t 8`；E2 ref；ratio 分母 |
| MyOS threads=8（刀⑦） | `s1-myos-bf16-t8` | **PASS** | **PASS** | 14.3503 | 9092.903 | 69.685 | 2963.383 | host 2026-11-03 15:02 UTC+8；`bf16` / `bf16_w_fp32_acc`，batch=1，`gemv_pack=false`；commit `9577e66`。卡：`arena/s1-myos-bf16-cost-card.md`。已合 main @ `a92677f1` |
| MyOS threads=8（刀⑧ prefill） | `s1-myos-bf16-t8-k8` | **PASS** | **PASS** | 14.9912 | 2193.653 | 66.706 | 2963.520 | host 2026-11-04 ≈07:57–07:59 UTC+8，WALL≈90s；`bf16` / `bf16_w_fp32_acc`，batch=1，`gemv_pack=false`；commit `16afd923`。卡：`arena/s1-myos-bf16-t8-k8-cost-card.md`。口径：`docs/phase1-knife7-verdict-and-knife8.md` |
| MyOS threads=8（刀⑨ decode） | `s1-myos-bf16-t8-k9` | **PASS** | **PASS** | 15.4880 | 1994.139 | 64.566 | 2963.445 | host 2026-11-04 ≈08:48–08:50 UTC+8；软件预取默认关；`gemv_pack=false`；commit `595943d`。T_decode 未达平带（要 ≥16.37/16.42），场景加权仍 −80。卡：`arena/s1-myos-bf16-t8-k9-cost-card.md`。口径：`docs/phase1-knife8-verdict-and-knife9.md` |
| MyOS threads=8（刀⑪ decode 融合） | `s1-myos-bf16-t8-k11` | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | 云端无 1.5B 权重，中位待 host。默认路径：Q/K/V 与 gate/up 各一次 same-x GEMV；`greedy_decode` 复用层权重视图。pack 关、预取关，无加速开关。对照刀⑨ T_decode **15.4880**。硬门：CORRECT+DET PASS；T_decode ≥16.42 或（≥16.37 且加权优于 −80）；L_TTFT ≤2193.6；相对刀⑨ T_decode ≥15.4880。卡：`arena/s1-myos-bf16-t8-k11-cost-card.md`。口径：`docs/phase1-knife10-verdict-and-knife11.md` |
| MyOS `42acbb39` | `e2-myos-s1` | （单次 greedy） | **PASS** exact 128/128 | — | — | — | — | 历史正确性收口；指纹 threads=1，**不**拿来和 `-t 8` 比吞吐 |
| llama.cpp b11960 | `s1-llamacpp-fp16-b11960` | PASS | （非 E2 ref） | 18.84 | 841.1 | 53.07 | 6377 | 官方 FP16；并列主赛道参考，非本 E2 金标准 |

胜负列（S1 BF16，刀⑧ MyOS threads=8 vs llama `-t 8`）：

| 指标 | MyOS | llama | ratio (MyOS/base) | 判定 |
|------|------|-------|-------------------|------|
| T_decode | 14.9912 | 17.2348 | 0.8698 | L |
| L_TTFT | 2193.653 | 608.110 | 3.6073 | L |
| L_TPOT | 66.706 | 58.022 | 1.1497 | L |
| M_peak | 2963.520 | 3050.953 | 0.9713 | T |
| 场景 | — | — | 加权和 -80 | **L**（CORRECT/DET 均 PASS，吞吐可比；TTFT 达标，场景分未优于 -80） |

刀⑦对照仍在上表刀⑦行，不因本表改写。权重：T_decode 0.35 / L_TPOT 0.25 / L_TTFT 0.20 / M_peak 0.20。这不是「已经胜过 llama」的对外句，也不把 TTFT 改善写成场景胜。软记录（GEMV 微基准，不是 S1）：gate/up `[8960, 1536]`，threads=1 → 8 为 3.611×（云端 4 cores）。见 `docs/phase1-knife7-closeout.md`。

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

1. 刀⑦已合 main @ `a92677f1`（PR #39）。账本刀⑦行与 `arena/s1-myos-bf16-cost-card.md` 是该收口的 host 中位。刀⑧补测不改这些数字。
2. 刀⑧硬闸 L_TTFT / T_decode / CORRECT / DET 已过：L_TTFT 2193.653 ≤ 3000，且相对刀⑦ 9092.903 ≈ 4.15×；T_decode 14.9912 ≥ 12.9；CORRECT=PASS、DET=PASS。场景加权和恰好 −80。口径写的是「优于 −80」；深析判此项未过（TTFT 达标、场景分未改善）。不对外宣称场景改善或场景胜。
3. 刀⑨ host 已填（`595943d`，指纹 `s1-myos-bf16-t8-k9`）：CORRECT/DET PASS；L_TTFT 1994.139 ≲ 2413；T_decode 15.4880（要 ≥16.42，或 ≥16.37 且加权优于 −80）**未过**；加权仍 −80。详见 `docs/phase1-knife9-verdict-and-knife10.md`（本地草稿）。不对外宣称 decode 平带或场景改善。不在本步开 S2/S3/S6，副赛道 Q4 保持关闭。
4. 刀⑪草稿（指纹 `s1-myos-bf16-t8-k11`）端到端 S1 **UNMEASURED**。粗路径是多投影融合（Q/K/V、gate/up）加跨 decode 步权重视图借用，不是行切、预取或 pack。对照基线是刀⑨ T_decode 15.4880。host 填中位后再判硬门。不把未测数字写成达标。
