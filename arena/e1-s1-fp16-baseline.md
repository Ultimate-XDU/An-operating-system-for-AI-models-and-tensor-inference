# E1 · llama.cpp S1 main-track (official FP16 GGUF)

**Date:** 2026-11-03 (UTC+8)  
**Fingerprint id / run dir:** `/workspace/MyOS/runs/s1-llamacpp-fp16-b11960`  
**Track:** `main-track-gguf-fp16` — official Hugging Face FP16 GGUF  
**Caveat:** MyOS is ST-BF16; **exact vs MyOS is NOT claimed** from this card. E2 will compare MyOS ST-BF16 vs this FP16 baseline and/or a BF16 GGUF later.

## Artifacts

| File | Role |
|------|------|
| `run_fingerprint.json` | §1.3 fingerprint |
| `tokens.txt` | common generated text (DET-PASS) |
| `token_ids.txt` | common next-token id list (128 ids) |
| `metrics.json` | W=3 discarded + R=5 scored; medians |
| `cost-card.md` | human summary |
| `det_evidence.json` | all scored texts + token id lists |

Raw per-run stderr/stdout/ids: `/tmp/e1-s1-fp16-raw/`

## Model / binary

| Item | Value |
|------|--------|
| GGUF | `/workspace/MyOS/weights/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-fp16.gguf` |
| SHA256 | `fc89e330deb3fd8fa560f1c0f35a1e2b8da96d59e13445559ed190307a6f5649` |
| Size | ≈ 3.4 GB (F16) |
| Binary | `/workspace/tools/llama.cpp-bin/llama-b11960/llama-cli` |
| Build | b11960 / commit `8642541b255bfb5dc39cd351a178be09013a55d6` |
| `LD_LIBRARY_PATH` | `/workspace/tools/llama.cpp-bin/llama-b11960` |

## Prompt

- File: `/workspace/MyOS/arena/prompts/s1_prompt.txt` — **user message body only** (UTF-8)
- llama-cli default Qwen jinja + system: `You are Qwen, created by Alibaba Cloud. You are a helpful assistant.`
- **Measured `prompt_n=128`** on all W+R runs (`slot print_timing`)

## Threads / cmdline

- Pinned **threads=8** (`nproc` once)
- Greedy clamps: `--temp 0 --top-k 1 --top-p 1.0 --min-p 0.0 --typical 1.0 --repeat-penalty 1.0 --presence-penalty 0 --frequency-penalty 0 --dry-multiplier 0 -s 0`
- See `run_fingerprint.json` → `cmdline`

## Medians (R=5)

| metric | median |
|--------|--------|
| T_decode (tok/s) | 18.8425 |
| L_TTFT / L_prefill (ms) | 841.080 |
| L_TPOT (ms) | 53.072 |
| M_peak (MiB RSS) | 6376.812 |
| T_e2e (tok/s) | 33.3481 |

Note: score1/score2 decode rates were depressed (~1–1.8 tok/s) while a concurrent Q4 DET probe shared the box; score3–5 settled ≈18.8–19.3 tok/s. Protocol median of R=5 → **18.84 tok/s**.

## DET

- **DET-PASS** at fingerprint `-t 8` across R=5: `ids_identical=True`, `text_identical=True`, lens=`[128,128,128,128,128]`, `first_diff=null`.
- Token ids from llama-cli `-v` stderr `next token: <id>` lines.
- No noamx / `-t 1` appendix needed (default `-t 8` already closed).

## Not claimed

- No MyOS comparison / no CORRECT gate vs MyOS (FP16 GGUF ≠ MyOS ST-BF16).
- E1 main-track baseline for this fingerprint is **closed** (DET-PASS); E2 exact remains dtype-caveated until BF16 GGUF or MyOS ST-BF16 pairing.
