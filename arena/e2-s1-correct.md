# E2 · S1 CORRECT 账本条目

**日期**：2026-11-03（UTC+8）  
**记录**：博弈手（协议账本，待 OSdeveloper 合入）  
**协议**：`PHASE1-ARENA-PROTOCOL-v0.1.md` §2.4 / §8 E2  
**索引**：总览 `arena/phase1-scoreboard.md` · E1 BF16 基线 `arena/e1-s1-bf16-baseline.md` · cost card `arena/e1-s1-bf16-cost-card.md` · 实现笔记 `arena/e2-s1-myos-exact.md`

## 结论

| 项 | 值 |
|----|-----|
| 场景 | S1（n_prompt=128, n_gen=128, B=1） |
| 赛道 | 主赛道 BF16（MyOS SafeTensors BF16 ↔ llama.cpp 自转 GGUF BF16） |
| **CORRECT** | **PASS**（exact 128/128） |
| DET（ref） | llama.cpp BF16 基线 DET-PASS（见 E1） |
| 吞吐对打 | **未开**（本条只收口正确性门闩） |
| 副赛道 Q4 | **仍未开对打** |

## 双方

| 角色 | 实现 | 路径 / 标识 |
|------|------|-------------|
| 参考 / 金标准 | llama.cpp b11960 | `runs/s1-llamacpp-bf16-b11960/`（`token_ids.txt`） |
| 挑战者 | MyOS main | commit `42acbb3920c597ab31e61f72e4cf90502b915584`（PR#27）；`runs/e2-myos-s1/` |

## 核对（本机复验）

- `diff`：`runs/s1-llamacpp-bf16-b11960/token_ids.txt` vs `runs/e2-myos-s1/token_ids.txt` → **identical**
- token 数：双方各 128
- 挑战者 fingerprint：`runs/e2-myos-s1/run_fingerprint.json`（fmt=safetensors, quant_or_dtype=bf16, threads=1, device=cpu, deterministic=true）
- 参考 fingerprint：`runs/s1-llamacpp-bf16-b11960/run_fingerprint.json`（fmt=gguf, bf16, threads=8, DET-PASS）

## 实现侧备注（来自 OSdeveloper，不作吞吐胜负）

- 墙钟约 11:50–12:21 UTC+8（≈31 min）；naive BF16 GEMV / threads=1
- **不**据此填写 T_decode / L_* 主表胜负；待正式 cost-card 对打任务再记

## 协议效力

- 主赛道 BF16 llama.cpp 基线继续作为 **exact 金标准**
- E2 门闩已过 → 允许后续开主赛道 S1 **吞吐/延迟/内存**对打（仍须双方 fingerprint 对齐规则）
- 副赛道 Q4：DET-FAIL 参考卡与 -det 附录维持原口径，**不对打**
