# Phase1 刀⑦口径 · S1 正式对打闭环（粗粒度）

| 项 | 内容 |
|----|------|
| 文档 | `docs/phase1-next-coarse-knife-7.md` |
| 前提 | ⑤收口；⑥负结果 @ `5bb4065f`；R18 PASS；scoreboard「主赛道 S1 吞吐对打」仍空；协议 §1.2 钉同一 `n_threads`，E1 llama S1 BF16 为 `-t 8` |
| 效力 | 与协议冲突时以 `PHASE1-ARENA-PROTOCOL-v0.1.md` 为准 |
| 粒度 | **一整块**：threads 对齐实现 + 计量/cost-card + S1 主表闭环。禁止拆成 RoPE/RMSNorm/单 GEMV 微刀 |

## 为何是这一条（不是融合微刀、不是再 pack）

1. 协议/账本下一步写死：正式 cost-card 对打；threads 不对齐则指纹不同 → **不可比**（§1.3）。
2. ⑥已证伪「同核运行时 pack」；单线程 ~80s vs llama ~8.8s（`-t 8`）——下一档主路径收益应来自 **可对打的并行宽度**，不是再抠标量核。
3. 月星要粒度放大：一口吃掉 **对打闭环**（可附带主路径并行收益），勿切小模块。

## 目标

打通主赛道 BF16 **S1 正式对打闭环**：MyOS 与 llama.cpp 在 **同一 `n_threads`** 指纹下出 cost-card（W=3 不计分 + R=5 中位），CORRECT 仍为胜负前置硬门；scoreboard S1 行补齐 T_decode / L_* / M_peak 与 W/T/L（或平）。

**不要求**本刀已达 ±5% 胜；要求 **可比** 且数字进主表。

## 主路径改动面（一整块，三件套同刀）

### A. Threads 对齐（主路径收益面）

- 对打默认指纹：**`threads=8`**，与现 E1 `s1-llamacpp-bf16-b11960` 对齐（禁止继续拿 `threads=1` 去填主胜负表冒充对打）。
- 实现：**GEMV 行维确定性并行**——行块分给 worker，**每行内部**仍 `j=0..k-1` 先乘后加、非 FMA、与 scalar `dot_row` 同比特；禁止跨行归约打乱单行序。
- 允许标准库线程（runtime 侧编队）或等价；**core 数值核**保持 no_std、可被多线程调用。不引入 rayon 若会破坏顺序/可复现；若用任务池须钉确定分割。
- CLI/指纹写入真实 `threads`；`batch` 仍为 1。

### B. 计量与 cost-card（闭环面）

- 计时边界按协议 §2.1：`t_load` 单列；`t_prefill` / `t_decode` 分离；产出 T_decode、L_TTFT/L_prefill、L_TPOT、M_peak。
- Runner：S1 上 W=3 丢弃 + R=5 中位；产物进 `runs/` + `arena/` cost-card（字段对齐既有 llama 卡）。
- DET：同指纹 R 次 token 序列一致（协议 §2.3 ON）。

### C. 账本

- 更新 `arena/phase1-scoreboard.md` 主赛道 S1：MyOS 正式行 + 与 llama BF16 的 ratio/verdict（按 §4.2）。
- 吞吐对打里程碑从 ⬜ 改为有数字（胜负可为负/平，但不得再空）。

## 硬验收

| # | 门 | 硬？ |
|---|----|------|
| 1 | 对打指纹 `threads` 与 llama S1 BF16 基线一致（**8**，除非月星改令重跑 E1 `-t 1`） | **硬** |
| 2 | 该指纹下 S1 CORRECT PASS（128/128 vs 金标准 token 文件） | **硬** |
| 3 | 同指纹 DET-PASS（R=5 ids 一致）或协议允许的单次+说明——主表默认要 DET | **硬** |
| 4 | 出具 MyOS S1 cost-card：W=3/R=5 中位 T_decode、L_TTFT、L_TPOT、M_peak | **硬** |
| 5 | scoreboard S1 MyOS 行非空，并与 llama 行可算 ratio | **硬** |
| 6 | 指纹仍 `bf16` / `bf16_w_fp32_acc`；无整表 f32；无默认开 pack | **硬** |
| 7 | 相对现状墙钟有可复核改善（期望来自行维并行；**不设 ≥2× 硬门**，避免⑥式假门——但对打数字必须落地） | 软（记录） |

## 不做清单

| 不做 | 原因 |
|------|------|
| 同族运行时二次 pack / 再改 tile 拷贝 | ⑥ 已负结果禁止 |
| 拆 RoPE、RMSNorm、SiLU 等单算子微刀 | 月星：粒度放大 |
| 先开 S2/S3 场景矩阵 | 先闭 S1 对打 |
| 副赛道 Q4 对打 | 仍不开 |
| 加载期整表 f32 / 整模二份常驻 | BF16 视图门闩 |
| 用 FMA 或打乱单行累加序换速度 | exact 硬门 |
| 本刀未出 cost-card 却宣称「已对打」或「已胜 llama」 | 闭环未完成 |

## 与⑤⑥边界

| | ⑤ | ⑥ | **⑦（本刀）** |
|--|----|----|----------------|
| 题 | 单核 SIMD/多行 | 运行时 pack 带宽（失败） | **threads 对齐 + S1 对打闭环** |
| 布局/并行 | 同布局单线程 SIMD | 二次拷贝 | **行维多线程；单行序不变** |
| 收口物 | ≥10× vs E2 + CORRECT | 负结果文档 | **cost-card + scoreboard 数字** |
| 互吞 | ⑦ 不重做 SIMD 行宽竞赛 | ⑦ 不复活 pack 主路径 | ⑤⑥ 不代替 cost-card |

## CloudAgent

**适合开干（一块 PR/一块 agent）**。Steer 须钉：

1. 对打 `threads=8` + 行维确定性并行 + 单行乘加序硬
2. CORRECT/DET 硬；先计量闭环，勿宣称 ±5% 胜
3. 禁 pack 默认开、禁整表 f32、禁拆成单算子微 PR 交差
4. PR 附：cost-card 路径、scoreboard 补丁、CORRECT/DET 证据、墙钟前后（软）
