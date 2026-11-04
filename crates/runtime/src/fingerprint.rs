//! Host-only `run_fingerprint.json` for one Phase1 process.
//!
//! This stays in `myos_runtime`. `model_core` does not open files, and it does
//! not learn the CLI. Fields we cannot know yet are JSON null. This writer
//! does not hash the weight file and does not copy the layout document's LFS
//! oid into `model_sha256`.

use std::fs;
use std::path::{Path, PathBuf};

use model_core::BlockShape;
use tensor_core::{ACC_PATH, QUANT_OR_DTYPE};

/// Which host decode just finished. The id is the directory under `runs/`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunKind {
    /// `--weights`, shape [`BlockShape::qwen25_1_5b`] when the file matches.
    Weights,
    /// `--block-fixture`, the in-process miniature block.
    BlockFixture,
}

impl RunKind {
    pub fn id(self) -> &'static str {
        match self {
            RunKind::Weights => "s1-weights",
            RunKind::BlockFixture => "s1-block-fixture",
        }
    }
}

/// `runs/<id>/`, relative to the process working directory.
pub fn default_run_dir(kind: RunKind) -> PathBuf {
    PathBuf::from("runs").join(kind.id())
}

/// Inputs for one fingerprint. Counts are what this process actually ran.
pub struct RunSpec {
    pub kind: RunKind,
    pub shape: BlockShape,
    /// First prompt id. The default weights path passes 0.
    pub prompt_token: u32,
    pub n_prompt: u64,
    pub steps: usize,
    pub cmdline: String,
    pub commit: Option<String>,
    /// Set when `--prompt-ids` supplied the prompt for this run.
    pub e2_prompt_ids: bool,
    /// CLI `--threads`. Showdown default is 8.
    pub threads: u64,
    /// Knife-6 panel pack. Showdown keeps this false.
    pub gemv_pack: bool,
    /// Replaces `kind.id()` when this process is the S1 showdown directory.
    pub run_id: Option<String>,
}

/// Protocol §1.3, plus the BF16 accumulate path this tree already pinned.
///
/// `model_sha256` and `seed` stay empty. `scenario_id` is `S1`. Without
/// `--prompt-ids`, `n_prompt` is 1. With `--prompt-ids`, `n_prompt` is the
/// file length and `n_gen` is `--steps` (protocol S1 is 128 and 128).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunFingerprint {
    pub model_id: Option<String>,
    pub model_sha256: Option<String>,
    pub fmt: String,
    pub quant_or_dtype: String,
    pub acc_path: String,
    pub n_ctx: u64,
    pub n_prompt: u64,
    pub n_gen: u64,
    pub batch: u64,
    pub threads: u64,
    /// Knife-6 runtime panel pack. Default showdown value is false.
    pub gemv_pack: bool,
    pub device: String,
    pub deterministic: bool,
    pub seed: Option<u64>,
    pub commit: Option<String>,
    pub cmdline: String,
    pub scenario_id: String,
    pub run_id: String,
    pub layout: String,
    pub num_layers: u64,
    pub hidden_size: u64,
    pub intermediate_size: u64,
    pub vocab_size: u64,
    pub prompt_token: u64,
    pub notes: String,
}

const NOTES: &str = "\
model_sha256 is null: this process does not hash the file and does not copy the layout doc LFS oid. \
seed is null: greedy argmax has no RNG. \
scenario_id S1 is a placeholder; n_prompt, n_gen, and n_ctx are this process, not protocol S1 (n_prompt=128, n_gen=128, n_ctx>=512). \
model_id is set only when layout is qwen25_1_5b. \
threads is the --threads value (showdown default 8, the same n_threads as llama S1 BF16 -t 8). batch is 1. \
gemv_pack is false unless MYOS_GEMV_PACK=1; knife-6 panel pack is not the default path. \
quant_or_dtype stays bf16 and acc_path stays bf16_w_fp32_acc (no whole-table f32). \
This file is not a llama.cpp exact comparison.";

const NOTES_E2: &str = "\
model_sha256 is null: this process does not hash the file and does not copy the layout doc LFS oid. \
seed is null: greedy argmax has no RNG. \
E2-capable multi-token path: --prompt-ids is prefilled as one chunk (projections and a causal KV write), then greedy --steps tokens are decoded with forward_token. \
n_prompt is that file's length and n_gen is --steps (protocol S1 is n_prompt=128, n_gen=128). \
n_ctx is n_prompt + n_gen, the KvCache max_seq for this process. \
model_id is set only when layout is qwen25_1_5b. \
threads is the --threads value (showdown default 8, the same n_threads as llama S1 BF16 -t 8). batch is 1. \
gemv_pack is false unless MYOS_GEMV_PACK=1; knife-6 panel pack is not the default path. \
quant_or_dtype stays bf16 and acc_path stays bf16_w_fp32_acc (no whole-table f32). \
--ref-ids prints CORRECT. This fingerprint file is not itself a llama.cpp exact comparison.";

impl RunFingerprint {
    pub fn from_spec(spec: RunSpec) -> Self {
        let pinned = spec.shape == BlockShape::qwen25_1_5b();
        let (model_id, fmt, layout) = match spec.kind {
            RunKind::Weights if pinned => (
                Some("Qwen/Qwen2.5-1.5B-Instruct".to_string()),
                "safetensors",
                "qwen25_1_5b",
            ),
            RunKind::Weights => (None, "safetensors", "not-qwen25-1.5b"),
            RunKind::BlockFixture => (None, "in-process", "block-fixture"),
        };
        let n_gen = spec.steps as u64;
        Self {
            model_id,
            model_sha256: None,
            fmt: fmt.to_string(),
            quant_or_dtype: QUANT_OR_DTYPE.to_string(),
            acc_path: ACC_PATH.to_string(),
            n_ctx: spec.n_prompt.saturating_add(n_gen),
            n_prompt: spec.n_prompt,
            n_gen,
            batch: 1,
            threads: spec.threads,
            gemv_pack: spec.gemv_pack,
            device: "cpu".to_string(),
            deterministic: true,
            seed: None,
            commit: spec.commit,
            cmdline: spec.cmdline,
            scenario_id: "S1".to_string(),
            run_id: spec.run_id.unwrap_or_else(|| spec.kind.id().to_string()),
            layout: layout.to_string(),
            num_layers: spec.shape.num_layers as u64,
            hidden_size: spec.shape.attn.hidden as u64,
            intermediate_size: spec.shape.intermediate as u64,
            vocab_size: spec.shape.vocab as u64,
            prompt_token: u64::from(spec.prompt_token),
            notes: if spec.e2_prompt_ids {
                NOTES_E2.to_string()
            } else {
                NOTES.to_string()
            },
        }
    }

    /// One JSON object, protocol fields first, then the fields this tree adds.
    /// A trailing newline is included so the file is a text file.
    pub fn to_json(&self) -> String {
        let fields = [
            ("model_id", json_opt_string(self.model_id.as_deref())),
            (
                "model_sha256",
                json_opt_string(self.model_sha256.as_deref()),
            ),
            ("fmt", json_string(&self.fmt)),
            ("quant_or_dtype", json_string(&self.quant_or_dtype)),
            ("acc_path", json_string(&self.acc_path)),
            ("n_ctx", self.n_ctx.to_string()),
            ("n_prompt", self.n_prompt.to_string()),
            ("n_gen", self.n_gen.to_string()),
            ("batch", self.batch.to_string()),
            ("threads", self.threads.to_string()),
            ("gemv_pack", bool_json(self.gemv_pack)),
            ("device", json_string(&self.device)),
            ("deterministic", bool_json(self.deterministic)),
            ("seed", json_opt_u64(self.seed)),
            ("commit", json_opt_string(self.commit.as_deref())),
            ("cmdline", json_string(&self.cmdline)),
            ("scenario_id", json_string(&self.scenario_id)),
            ("run_id", json_string(&self.run_id)),
            ("layout", json_string(&self.layout)),
            ("num_layers", self.num_layers.to_string()),
            ("hidden_size", self.hidden_size.to_string()),
            ("intermediate_size", self.intermediate_size.to_string()),
            ("vocab_size", self.vocab_size.to_string()),
            ("prompt_token", self.prompt_token.to_string()),
            ("notes", json_string(&self.notes)),
        ];
        let mut out = String::from("{\n");
        for (i, (key, value)) in fields.iter().enumerate() {
            out.push_str("  \"");
            out.push_str(key);
            out.push_str("\": ");
            out.push_str(value);
            if i + 1 != fields.len() {
                out.push(',');
            }
            out.push('\n');
        }
        out.push_str("}\n");
        out
    }
}

/// Create `dir` and write `dir/run_fingerprint.json`.
pub fn write_run_fingerprint(dir: &Path, fp: &RunFingerprint) -> std::io::Result<PathBuf> {
    fs::create_dir_all(dir)?;
    let path = dir.join("run_fingerprint.json");
    fs::write(&path, fp.to_json())?;
    Ok(path)
}

/// Write generated token ids as space-separated `u32`s, one trailing newline.
pub fn write_token_ids(dir: &Path, ids: &[u32]) -> std::io::Result<PathBuf> {
    fs::create_dir_all(dir)?;
    let path = dir.join("token_ids.txt");
    let mut text = String::new();
    for (i, id) in ids.iter().enumerate() {
        if i > 0 {
            text.push(' ');
        }
        text.push_str(&id.to_string());
    }
    text.push('\n');
    fs::write(&path, text)?;
    Ok(path)
}

/// Resolve `HEAD` by walking parents for a `.git` directory or gitfile.
///
/// Returns `None` when this process cannot see a repository. A missing git
/// directory is not a failed run.
pub fn git_commit_at(start: &Path) -> Option<String> {
    let mut current = Some(start);
    while let Some(dir) = current {
        let git = dir.join(".git");
        if git.exists() {
            return read_git_head(&git);
        }
        current = dir.parent();
    }
    None
}

fn read_git_head(git: &Path) -> Option<String> {
    let git_dir = if git.is_dir() {
        git.to_path_buf()
    } else {
        let text = fs::read_to_string(git).ok()?;
        let line = text.lines().next()?.trim();
        let rel = line.strip_prefix("gitdir:")?.trim();
        let path = PathBuf::from(rel);
        if path.is_absolute() {
            path
        } else {
            git.parent()?.join(path)
        }
    };
    let head = fs::read_to_string(git_dir.join("HEAD")).ok()?;
    let head = head.trim();
    if let Some(rest) = head.strip_prefix("ref: ") {
        let name = rest.trim();
        let ref_file = git_dir.join(name);
        if let Ok(sha) = fs::read_to_string(&ref_file) {
            let sha = sha.trim();
            if is_full_sha(sha) {
                return Some(sha.to_string());
            }
        }
        return packed_ref(&git_dir, name);
    }
    if is_full_sha(head) {
        Some(head.to_string())
    } else {
        None
    }
}

fn packed_ref(git_dir: &Path, name: &str) -> Option<String> {
    let text = fs::read_to_string(git_dir.join("packed-refs")).ok()?;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with('^') {
            continue;
        }
        let mut parts = line.split_whitespace();
        let sha = match parts.next() {
            Some(sha) => sha,
            None => continue,
        };
        let r = match parts.next() {
            Some(r) => r,
            None => continue,
        };
        if r == name && is_full_sha(sha) {
            return Some(sha.to_string());
        }
    }
    None
}

fn is_full_sha(s: &str) -> bool {
    s.len() == 40 && s.bytes().all(|b| b.is_ascii_hexdigit())
}

fn bool_json(value: bool) -> String {
    if value {
        "true".to_string()
    } else {
        "false".to_string()
    }
}

fn json_opt_u64(value: Option<u64>) -> String {
    match value {
        Some(n) => n.to_string(),
        None => "null".to_string(),
    }
}

fn json_opt_string(value: Option<&str>) -> String {
    match value {
        Some(text) => json_string(text),
        None => "null".to_string(),
    }
}

fn json_string(text: &str) -> String {
    let mut out = String::from("\"");
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}
