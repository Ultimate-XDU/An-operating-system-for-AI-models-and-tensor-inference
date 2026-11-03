# E1 llama.cpp setup (box)

Date: 2026-11-03 (UTC+8)

## Binary

| Item | Value |
|------|--------|
| **Binary path** | `/workspace/tools/llama.cpp-bin/llama-b11960/llama-cli` |
| **Convenience symlink** | `/workspace/tools/llama-cli-bin/llama-cli` → same dir |
| **Variant** | Official release **CPU-only** Ubuntu x64 (no CUDA) |
| **Release tag** | `b11960` |
| **Build / commit** | `version: 0.8.0-dev (build 11960, commit 8642541b2)` → full SHA `8642541b255bfb5dc39cd351a178be09013a55d6` |
| **Asset** | `https://github.com/ggml-org/llama.cpp/releases/download/b11960/llama-b11960-bin-ubuntu-x64.tar.gz` |
| **Runtime** | Set `LD_LIBRARY_PATH=/workspace/tools/llama.cpp-bin/llama-b11960:$LD_LIBRARY_PATH` (shared libs live next to the binary) |

Source checkout (for convert scripts / docs; **not** the built CLI used below):

- Path: `/workspace/tools/llama.cpp`
- Tip SHA at clone time: `2cbca389f515df455412dafa07e2b1359cc72a6d` (master shallow clone)

## Why prebuilt (not local cmake build)

- Box apt mirrors (`deb.debian.org`) returned 500 / SSL EOF; could not `apt install g++ cmake`.
- Portable Kitware CMake **3.31.6** installed at `/workspace/tools/cmake/bin/cmake`.
- `make` installed via direct `.deb`.
- **No `g++`** (only `gcc` present); full C++ toolchain debs have a deep Depends graph (`g++-14-x86-64-linux-gnu`, etc.) — left for a later apt-healthy session.
- Makefile in-tree only errors: build system is CMake-only now.

When apt works: from `/workspace/tools/llama.cpp`:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF
cmake --build build -j$(nproc) --target llama-cli
# expect: build/bin/llama-cli
```

## GGUF used for smoke

- Path: `/workspace/MyOS/weights/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-q4_k_m.gguf`
- SHA256: `6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e`
- Size ≈ 1.12 GB (Q4_K_M)

## Working smoke command (CPU greedy, 4 tokens)

Flags match this build’s `--help` (no renames vs the planned CLI; `--single-turn` added so the process exits after one reply instead of staying in chat UI).

```bash
export LD_LIBRARY_PATH=/workspace/tools/llama.cpp-bin/llama-b11960:$LD_LIBRARY_PATH
BIN=/workspace/tools/llama.cpp-bin/llama-b11960/llama-cli
GGUF=/workspace/MyOS/weights/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-q4_k_m.gguf

$BIN -m "$GGUF" \
  -ngl 0 -t $(nproc) -c 512 -n 4 \
  --temp 0 --top-k 1 --repeat-penalty 1.0 -s 0 \
  -p "Hi" --no-display-prompt --single-turn
```

### Observed smoke output

- Build line: `b11960-8642541b2`
- ftype: `Q4_K - Medium`
- Generated text (≈4 tokens): **`Hello! How can`**
- Exit code: 0
- Rough timings (box, 8 threads): prompt ~300+ t/s, generation ~20–30 t/s

## Flag notes for E1 (this build)

| Intent | Flag (b11960) | Notes |
|--------|----------------|-------|
| CPU-only | `-ngl 0` | still present (`--gpu-layers`) |
| threads | `-t N` | |
| ctx | `-c N` | |
| n_gen | `-n N` / `--n-predict` | |
| greedy | `--temp 0` **and** `--top-k 1` | both available under “sampling params” |
| no repeat pen | `--repeat-penalty 1.0` | |
| seed | `-s 0` | |
| hide prompt | `--no-display-prompt` | |
| one-shot exit | `--single-turn` / `-st` | recommended for scripts; without it CLI stays in chat REPL after `-n` |
| avoid experimental backend sampler | do **not** pass `-bs` / `--backend-sampling` | |

No sampler flag renames discovered relative to the E1 checklist; names above are current.

## Not done here

- Local compile of tip-of-tree `llama-cli` (blocked on g++)
- E2 MyOS ST-BF16 challenger run / CORRECT gate

## S1 Q4 side-track baseline (done)

- Run dir: [`/workspace/MyOS/runs/s1-llamacpp-q4km-b11960`](../runs/s1-llamacpp-q4km-b11960/)
- Details: [`arena/e1-s1-q4-baseline.md`](e1-s1-q4-baseline.md)
- Prompt: [`arena/prompts/s1_prompt.txt`](prompts/s1_prompt.txt) (user body; CLI chat-template → **n_prompt=128**)

## S1 Q4 DET follow-up

- DET-FAIL evidence: `runs/s1-llamacpp-q4km-b11960` (keep)
- DET-PASS (t=1, no-AMX backend dir): `runs/s1-llamacpp-q4km-b11960-det`
- Probe notes: `arena/e1-s1-q4-det-probe.md`

## S1 FP16 main-track baseline (done)

- Run dir: [`/workspace/MyOS/runs/s1-llamacpp-fp16-b11960`](../runs/s1-llamacpp-fp16-b11960/)
- Details: [`arena/e1-s1-fp16-baseline.md`](e1-s1-fp16-baseline.md)
- GGUF: `/workspace/MyOS/weights/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-fp16.gguf`
- SHA256: `fc89e330deb3fd8fa560f1c0f35a1e2b8da96d59e13445559ed190307a6f5649`
- Track: `main-track-gguf-fp16` · threads=8 · **DET-PASS** · measured n_prompt=128
- Medians: T_decode≈18.84 tok/s, L_TTFT≈841 ms, L_TPOT≈53.1 ms, M_peak≈6377 MiB
- Caveat: official FP16 GGUF; MyOS ST-BF16 exact not claimed here (E2)

## S1 BF16 main-track baseline (done · PREFERRED for MyOS exact)

- Run dir: [`/workspace/MyOS/runs/s1-llamacpp-bf16-b11960`](../runs/s1-llamacpp-bf16-b11960/)
- Details: [`arena/e1-s1-bf16-baseline.md`](e1-s1-bf16-baseline.md)
- GGUF: `/workspace/MyOS/weights/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-bf16.gguf`
- SHA256: `1a32a39cefefe73f82b4600ccecfe4f3f9411337543a7481f1aa3330d71c973b`
- Source: self-converted `convert_hf_to_gguf.py --outtype bf16` from HF weights under `weights/Qwen2.5-1.5B-Instruct`
- Track: `main-track-gguf-bf16` · stock b11960 · threads=8 · **DET-PASS** · measured n_prompt=128
- Medians: T_decode≈17.23 tok/s, L_TTFT≈608 ms, L_TPOT≈58.0 ms, M_peak≈3051 MiB
- No noamx / `-t 1` fallback needed
- Role: **preferred** llama baseline for later E2 exact vs MyOS ST-BF16
