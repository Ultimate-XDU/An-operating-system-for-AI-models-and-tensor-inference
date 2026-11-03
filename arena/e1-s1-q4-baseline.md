# E1 · llama.cpp S1 side-track (Q4_K_M)

**Date:** 2026-11-03 (UTC+8)  
**Track:** GGUF Q4_K_M **side-track only** (not MyOS vs llama exact; not main-track FP16/BF16)

## Runs

| Run dir | Role | DET | Notes |
|---------|------|-----|-------|
| [`runs/s1-llamacpp-q4km-b11960`](../runs/s1-llamacpp-q4km-b11960/) | first baseline (-t 8, default AMX) | **FAIL** | keep as evidence; cost = 吞吐/延迟/内存参考 · 胜负与 CORRECT 作废；**不进**副赛道主胜负表 |
| [`runs/s1-llamacpp-q4km-b11960-det`](../runs/s1-llamacpp-q4km-b11960-det/) | DET-stable (-t 1, no sapphirerapids / no AMX + env) | **PASS** | W=3 R=5; single-thread non-AMX throughput — appendix/reference until protocol accepts fingerprint for side-track baseline |

Probe write-up: [`e1-s1-q4-det-probe.md`](e1-s1-q4-det-probe.md)

## Prompt

- `/workspace/MyOS/arena/prompts/s1_prompt.txt` — user body only  
- CLI default Qwen jinja (+ system “You are Qwen…”) → **measured `n_prompt=128`**

## Status (博弈手口径)

- 吞吐已采（两张卡都有 medians）  
- DET：默认卡未过；稳定卡已过  
- 副赛道主胜负 / 对打：**未闭合、未开对打**，直至协议确认 DET-stable fingerprint  
