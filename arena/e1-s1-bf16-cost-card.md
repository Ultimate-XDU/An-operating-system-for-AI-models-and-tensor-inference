# Cost Card · S1 · 2026-11-03 11:41 CST (UTC+8)

### 设定
- fingerprint: `s1-llamacpp-bf16-b11960`
- model: Qwen2.5-1.5B-Instruct
- track: **self-converted GGUF BF16 (MAIN TRACK · PREFERRED for MyOS exact)**
- device: cpu, n_threads: 8
- baseline: llama.cpp b11960 / commit `8642541b255bfb5dc39cd351a178be09013a55d6`
- binary: `/workspace/tools/llama.cpp-bin/llama-b11960/llama-cli`
- MyOS challenger: **not run** (llama-only E1 baseline)
- prompt: `/workspace/MyOS/arena/prompts/s1_prompt.txt` (user body; CLI Qwen chat template → measured n_prompt=128)
- n_ctx=512, n_gen=128, batch=1, deterministic=ON, seed=0
- protocol: W=3 discard, R=5 median
- GGUF sha256: `1a32a39cefefe73f82b4600ccecfe4f3f9411337543a7481f1aa3330d71c973b` (convert_hf_to_gguf.py --outtype bf16)
- extra_env: `{}`

### Exact cmdline
```
LD_LIBRARY_PATH=/workspace/tools/llama.cpp-bin/llama-b11960 \

/workspace/tools/llama.cpp-bin/llama-b11960/llama-cli -m /workspace/MyOS/weights/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-bf16.gguf -ngl 0 -t 8 -c 512 -n 128 --temp 0 --top-k 1 --top-p 1.0 --min-p 0.0 --typical 1.0 --repeat-penalty 1.0 --presence-penalty 0.0 --frequency-penalty 0.0 --dry-multiplier 0.0 -s 0 -f /workspace/MyOS/arena/prompts/s1_prompt.txt --no-display-prompt --single-turn --no-warmup -v
```

### 结果（中位数，llama.cpp only）
| metric | llama.cpp | MyOS | ratio | verdict |
|--------|-----------|------|-------|---------|
| T_decode (tok/s) | 17.2348 | — | — | usable preferred BF16 main-track baseline (E2 exact vs MyOS ST-BF16) |
| L_TTFT / L_prefill (ms) | 608.110 | — | — | usable preferred BF16 main-track baseline (E2 exact vs MyOS ST-BF16) |
| L_TPOT (ms) | 58.022 | — | — | usable preferred BF16 main-track baseline (E2 exact vs MyOS ST-BF16) |
| M_peak (MiB RSS) | 3050.953 | — | — | usable preferred BF16 main-track baseline (E2 exact vs MyOS ST-BF16) |
| T_e2e (tok/s) | 31.8608 | — | — | appendix |

### DET
- **PASS**: DET-PASS ids_identical=True text_identical=True lens=[128, 128, 128, 128, 128]
- Evidence: `det_evidence.json`, per-run ids under `/tmp/e1-s1-bf16-raw/`
- CORRECT: **PASS** (E2 2026-11-03; MyOS `42acbb39` / `runs/e2-myos-s1` exact 128/128; ledger `arena/e2-s1-correct.md`)

### Generated text (score1; common only if DET-PASS)
```
The water cycle is like a big dance where water moves between the air and the land. First, water from rivers, lakes, and oceans evaporates into the air. This is like when you see steam rising from a pot of boiling water. The sun's heat makes water turn into tiny water vapor particles. These particles float in the air, like clouds. When the water vapor gets cold, it turns back into liquid water, like rain or snow. This is called precipitation. The rain or snow falls to the ground and collects in rivers, lakes, and oceans. Plants in forests and grasslands around rivers also help by releasing water through
```

### Notes
- Preferred main-track BF16 GGUF for later E2 exact vs MyOS ST-BF16 (same dtype family).
- Timings from llama.cpp `slot print_timing` (prompt eval / eval); M_peak = max VmRSS sampled from `/proc/<pid>/status`.
- CLI internal empty warmup off (`--no-warmup`); protocol warm-ups are full discarded runs.
- Token ids parsed from `-v` stderr `next token:` lines (same method as FP16/Q4 runs).
- DET-PASS: E1 main-track BF16 baseline closed for later E2 exact vs MyOS.
