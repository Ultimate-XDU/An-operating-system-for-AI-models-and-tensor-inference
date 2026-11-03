//! Spawn `myos` and check the host weights path without a 1.5B download.
use model_core::block_fixture;
use model_core::qwen25_layout::{layer_tensor_keys, EMBED_TOKENS, FINAL_NORM};
use safetensors::tensor::TensorView;
use safetensors::Dtype;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};
use weight_loader::WeightMap;

fn tiny_safetensors() -> Vec<u8> {
    let w = 1.0f32.to_le_bytes();
    let bias = 0.5f32.to_le_bytes();
    let w_view = TensorView::new(Dtype::F32, vec![1], &w).unwrap();
    let bias_view = TensorView::new(Dtype::F32, vec![], &bias).unwrap();
    safetensors::serialize([("w", w_view), ("bias", bias_view)], None).unwrap()
}

/// Same tensors as `block_fixture`, written as SafeTensors BF16. Not 1.5B.
fn miniature_block_safetensors() -> Vec<u8> {
    let map = block_fixture().unwrap();
    let mut names = vec![EMBED_TOKENS.to_string(), FINAL_NORM.to_string()];
    names.extend(layer_tensor_keys(0));
    let mut owned = Vec::new();
    for name in &names {
        let info = map.tensor_info(name).unwrap();
        let bytes = map.tensor_bytes(name).unwrap().to_vec();
        owned.push((name.clone(), info.shape.clone(), bytes));
    }
    let views: Vec<(&str, TensorView<'_>)> = owned
        .iter()
        .map(|(name, shape, bytes)| {
            (
                name.as_str(),
                TensorView::new(Dtype::BF16, shape.clone(), bytes).unwrap(),
            )
        })
        .collect();
    safetensors::serialize(views, None).unwrap()
}

struct TempFile(PathBuf);

impl TempFile {
    fn write(label: &str, bytes: &[u8]) -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "myos-runtime-{label}-{}-{nanos}.safetensors",
            std::process::id()
        ));
        std::fs::write(&path, bytes).unwrap();
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Each spawn gets its own cwd so parallel tests do not share `runs/`.
struct Scratch(PathBuf);

impl Scratch {
    fn new(label: &str) -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "myos-runtime-{label}-{}-{nanos}",
            std::process::id()
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn myos(dir: &Path) -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_myos"));
    cmd.env_remove("MYOS_WEIGHTS");
    cmd.current_dir(dir);
    cmd
}

fn stdout(cmd: &mut Command) -> (bool, i32, String, String) {
    let output = cmd.output().unwrap();
    let code = output.status.code().unwrap_or(1);
    let out = String::from_utf8(output.stdout).unwrap();
    let err = String::from_utf8(output.stderr).unwrap();
    (output.status.success(), code, out, err)
}

fn assert_bf16_fingerprint(json: &str) {
    assert!(json.contains("\"quant_or_dtype\": \"bf16\""), "{json}");
    assert!(json.contains("\"acc_path\": \"bf16_w_fp32_acc\""), "{json}");
    assert!(json.contains("\"deterministic\": true"), "{json}");
    assert!(json.contains("\"device\": \"cpu\""), "{json}");
    assert!(json.contains("\"model_sha256\": null"), "{json}");
    assert!(json.contains("\"seed\": null"), "{json}");
    assert!(json.contains("\"scenario_id\": \"S1\""), "{json}");
    assert!(json.contains("\"threads\": 8"), "{json}");
    assert!(json.contains("\"gemv_pack\": false"), "{json}");
    assert!(json.contains("\"batch\": 1"), "{json}");
}

#[test]
fn default_smoke_is_the_cycle() {
    let dir = Scratch::new("smoke");
    let (ok, code, out, err) = stdout(&mut myos(dir.path()));
    assert!(ok, "stderr={err}");
    assert_eq!(code, 0);
    assert_eq!(err, "");
    assert_eq!(out, "MyOS Phase1 smoke decode (prompt=0): [1, 2, 3, 0]\n");
    assert!(!dir.path().join("runs").exists());
}

#[test]
fn steps_flag_shortens_the_cycle_and_still_skips_the_fingerprint() {
    let dir = Scratch::new("steps-smoke");
    let (ok, _, out, err) = stdout(myos(dir.path()).args(["--steps", "2"]));
    assert!(ok, "stderr={err}");
    assert_eq!(out, "MyOS Phase1 smoke decode (prompt=0): [1, 2]\n");
    assert!(!dir.path().join("runs").exists());
}

#[test]
fn weights_flag_does_not_fall_through_to_smoke() {
    let dir = Scratch::new("flag");
    let tmp = TempFile::write("flag", &tiny_safetensors());
    let (ok, code, out, err) =
        stdout(myos(dir.path()).args(["--weights", tmp.path().to_str().unwrap()]));
    assert!(!ok, "stdout={out}");
    assert_eq!(code, 1);
    assert!(err.contains("MyOS weights"), "{err}");
    assert!(err.contains("tensor not found"), "{err}");
    assert!(!out.contains("smoke decode"), "{out}");
    assert!(!out.contains("[1, 2, 3, 0]"), "{out}");
    assert!(!out.contains("weights decode"), "{out}");
    assert!(!dir.path().join("runs").exists());
}

#[test]
fn miniature_bf16_file_is_rejected_by_the_1_5b_shape() {
    let dir = Scratch::new("mini");
    let tmp = TempFile::write("mini", &miniature_block_safetensors());
    let (ok, code, out, err) =
        stdout(myos(dir.path()).args(["--weights", tmp.path().to_str().unwrap()]));
    assert!(!ok, "stdout={out} stderr={err}");
    assert_eq!(code, 1);
    assert!(
        err.contains("tensor shape does not match the layout"),
        "{err}"
    );
    assert!(!out.contains("smoke decode"), "{out}");
    assert!(!out.contains("weights decode"), "{out}");
    assert!(!out.contains("[1, 1, 1, 1]"), "{out}");
    assert!(!dir.path().join("runs").exists());
}

#[test]
fn env_var_selects_weights_and_flag_overrides_it() {
    let dir = Scratch::new("env");
    let good = TempFile::write("env-good", &tiny_safetensors());
    let (ok, _, out, err) = stdout(myos(dir.path()).env("MYOS_WEIGHTS", good.path()));
    assert!(!ok, "stdout={out}");
    assert!(err.contains(good.path().to_str().unwrap()), "{err}");
    assert!(!out.contains("smoke decode"), "{out}");

    let missing = std::env::temp_dir().join(format!(
        "myos-runtime-missing-{}-{}.safetensors",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let (ok, _, out, err) = stdout(
        myos(dir.path())
            .env("MYOS_WEIGHTS", &missing)
            .args(["--weights", good.path().to_str().unwrap()]),
    );
    assert!(!ok, "flag should win over a missing env path; stdout={out}");
    assert!(err.contains(good.path().to_str().unwrap()), "{err}");
    assert!(!err.contains(&missing.display().to_string()), "{err}");
    assert!(!out.contains("smoke decode"), "{out}");
}

#[test]
fn bad_weights_do_not_print_smoke() {
    let dir = Scratch::new("bad");
    let missing = std::env::temp_dir().join(format!(
        "myos-runtime-absent-{}-{}.safetensors",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let (ok, code, out, err) =
        stdout(myos(dir.path()).args(["--weights", missing.to_str().unwrap()]));
    assert!(!ok);
    assert_eq!(code, 1);
    assert!(err.contains("MyOS weights"), "{err}");
    assert!(!out.contains("smoke decode"), "{out}");

    let garbage = TempFile::write("garbage", b"not-a-safetensors-file!!!!");
    let (ok, code, out, err) =
        stdout(myos(dir.path()).args(["--weights", garbage.path().to_str().unwrap()]));
    assert!(!ok);
    assert_eq!(code, 1);
    assert!(err.contains("could not be parsed"), "{err}");
    assert!(!out.contains("smoke decode"), "{out}");
}

#[test]
fn block_fixture_flag_is_not_the_cycle_smoke() {
    let dir = Scratch::new("fixture");
    let (ok, code, out, err) = stdout(myos(dir.path()).arg("--block-fixture"));
    assert!(ok, "stderr={err}");
    assert_eq!(code, 0);
    assert_eq!(err, "");
    assert_eq!(out, "MyOS Phase1 block fixture (prompt=0): [1, 1, 1, 1]\n");
    assert!(!out.contains("[1, 2, 3, 0]"));
    assert!(!out.contains("smoke decode"));
    let json = std::fs::read_to_string(
        dir.path()
            .join("runs/s1-block-fixture/run_fingerprint.json"),
    )
    .unwrap();
    assert_bf16_fingerprint(&json);
    assert!(json.contains("\"layout\": \"block-fixture\""), "{json}");
    assert!(json.contains("\"model_id\": null"), "{json}");
    assert!(json.contains("\"fmt\": \"in-process\""), "{json}");
    assert!(json.contains("\"n_gen\": 4"), "{json}");
    assert!(json.contains("\"n_ctx\": 5"), "{json}");
    assert!(json.contains("\"commit\": null"), "{json}");
    assert!(json.contains("\"n_prompt\": 1"), "{json}");
    assert!(!json.contains("E2-capable multi-token path"), "{json}");
    assert!(!dir
        .path()
        .join("runs/s1-block-fixture/token_ids.txt")
        .exists());
}

#[test]
fn block_fixture_twice_prints_the_same_tokens() {
    let a = Scratch::new("twice-a");
    let b = Scratch::new("twice-b");
    let (ok_a, _, out_a, err_a) = stdout(myos(a.path()).arg("--block-fixture"));
    let (ok_b, _, out_b, err_b) = stdout(myos(b.path()).arg("--block-fixture"));
    assert!(ok_a, "{err_a}");
    assert!(ok_b, "{err_b}");
    assert_eq!(out_a, out_b);
    assert_eq!(
        out_a,
        "MyOS Phase1 block fixture (prompt=0): [1, 1, 1, 1]\n"
    );
}

#[test]
fn fingerprint_dir_flag_overrides_the_default_runs_path() {
    let dir = Scratch::new("fp-dir");
    let dest = dir.path().join("out");
    let (ok, _, out, err) = stdout(myos(dir.path()).args([
        "--block-fixture",
        "--steps",
        "2",
        "--fingerprint-dir",
        dest.to_str().unwrap(),
    ]));
    assert!(ok, "stderr={err} stdout={out}");
    assert_eq!(out, "MyOS Phase1 block fixture (prompt=0): [1, 1]\n");
    let json = std::fs::read_to_string(dest.join("run_fingerprint.json")).unwrap();
    assert_bf16_fingerprint(&json);
    assert!(json.contains("\"n_gen\": 2"), "{json}");
    assert!(json.contains("\"n_ctx\": 3"), "{json}");
    assert!(!dir.path().join("runs").exists());
}

#[test]
fn weights_then_block_fixture_skips_cycle_smoke() {
    let dir = Scratch::new("both");
    let tmp = TempFile::write("flag-block", &tiny_safetensors());
    let (ok, _, out, err) = stdout(myos(dir.path()).args([
        "--weights",
        tmp.path().to_str().unwrap(),
        "--block-fixture",
    ]));
    assert!(ok, "stderr={err} stdout={out}");
    assert_eq!(out, "MyOS Phase1 block fixture (prompt=0): [1, 1, 1, 1]\n");
    assert!(!out.contains("smoke decode"));
    assert!(!out.contains("[1, 2, 3, 0]"));
    assert!(!out.contains("weights decode"));
    let json = std::fs::read_to_string(
        dir.path()
            .join("runs/s1-block-fixture/run_fingerprint.json"),
    )
    .unwrap();
    assert!(json.contains("\"layout\": \"block-fixture\""), "{json}");
    assert!(!dir.path().join("runs/s1-weights").exists());
}

#[test]
fn bad_weights_skip_the_block_fixture() {
    let dir = Scratch::new("bad-block");
    let missing = std::env::temp_dir().join(format!(
        "myos-runtime-block-missing-{}-{}.safetensors",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let (ok, code, out, err) =
        stdout(myos(dir.path()).args(["--block-fixture", "--weights", missing.to_str().unwrap()]));
    assert!(!ok);
    assert_eq!(code, 1);
    assert!(err.contains("MyOS weights"), "{err}");
    assert!(!out.contains("block fixture"), "{out}");
    assert!(!out.contains("smoke decode"), "{out}");
}

#[test]
fn unknown_argument_is_usage() {
    let dir = Scratch::new("usage");
    let (ok, code, out, err) = stdout(myos(dir.path()).arg("--bogus"));
    assert!(!ok);
    assert_eq!(code, 2);
    assert!(err.contains("unknown argument"));
    assert!(err.contains(
        "usage: myos [--weights PATH] [--block-fixture] [--steps N] [--threads N] [--fingerprint-dir DIR] [--prompt-ids PATH] [--ref-ids PATH] [--arena-s1] [--warmup N] [--repeats N] [--cost-card PATH]"
    ));
    assert_eq!(out, "");
}

#[test]
fn prompt_ids_without_weights_is_usage() {
    let dir = Scratch::new("prompt-usage");
    let (ok, code, out, err) = stdout(myos(dir.path()).args(["--prompt-ids", "ids.txt"]));
    assert!(!ok);
    assert_eq!(code, 2);
    assert!(err.contains("--prompt-ids needs --weights"), "{err}");
    assert_eq!(out, "");

    let (ok, code, out, err) = stdout(myos(dir.path()).args([
        "--block-fixture",
        "--weights",
        "a.safetensors",
        "--prompt-ids",
        "ids.txt",
    ]));
    assert!(!ok);
    assert_eq!(code, 2);
    assert!(
        err.contains("--prompt-ids cannot be combined with --block-fixture"),
        "{err}"
    );
    assert_eq!(out, "");
    assert!(!dir.path().join("runs").exists());
}

#[test]
fn empty_prompt_ids_errors_before_decode() {
    let dir = Scratch::new("empty-prompt");
    let tmp = TempFile::write("empty-prompt", &tiny_safetensors());
    let ids = dir.path().join("empty.txt");
    std::fs::write(&ids, " \n").unwrap();
    let (ok, code, out, err) = stdout(myos(dir.path()).args([
        "--weights",
        tmp.path().to_str().unwrap(),
        "--prompt-ids",
        ids.to_str().unwrap(),
    ]));
    assert!(!ok);
    assert_eq!(code, 1);
    assert!(err.contains("MyOS prompt-ids"), "{err}");
    assert!(err.contains("empty"), "{err}");
    assert!(!out.contains("smoke decode"), "{out}");
    assert!(!out.contains("weights decode"), "{out}");
    assert!(!dir.path().join("runs").exists());
}

#[test]
fn prompt_ids_on_tiny_weights_still_rejects_the_1_5b_shape() {
    let dir = Scratch::new("prompt-mini");
    let tmp = TempFile::write("prompt-mini", &miniature_block_safetensors());
    let ids = dir.path().join("ids.txt");
    std::fs::write(&ids, "0 1 2\n").unwrap();
    let (ok, code, out, err) = stdout(myos(dir.path()).args([
        "--weights",
        tmp.path().to_str().unwrap(),
        "--prompt-ids",
        ids.to_str().unwrap(),
        "--steps",
        "2",
    ]));
    assert!(!ok, "stdout={out}");
    assert_eq!(code, 1);
    assert!(
        err.contains("tensor shape does not match the layout"),
        "{err}"
    );
    assert!(!out.contains("weights decode"), "{out}");
    assert!(!out.contains("smoke decode"), "{out}");
    assert!(!dir.path().join("runs").exists());
}

#[test]
fn ref_ids_on_block_fixture_prints_correct_and_exits_zero() {
    let dir = Scratch::new("ref");
    let pass = dir.path().join("pass.txt");
    std::fs::write(&pass, "1 1 1 1\n").unwrap();
    let (ok, code, out, err) =
        stdout(myos(dir.path()).args(["--block-fixture", "--ref-ids", pass.to_str().unwrap()]));
    assert!(ok, "stderr={err} stdout={out}");
    assert_eq!(code, 0);
    assert_eq!(
        out,
        "MyOS Phase1 block fixture (prompt=0): [1, 1, 1, 1]\nCORRECT: PASS\n"
    );

    let fail = dir.path().join("fail.txt");
    std::fs::write(&fail, "1 2 1 1\n").unwrap();
    let (ok, code, out, err) =
        stdout(myos(dir.path()).args(["--block-fixture", "--ref-ids", fail.to_str().unwrap()]));
    assert!(ok, "stderr={err} stdout={out}");
    assert_eq!(code, 0);
    assert!(
        out.contains("CORRECT: FAIL first_diff@1: got=1 ref=2"),
        "{out}"
    );

    let short = dir.path().join("short.txt");
    std::fs::write(&short, "1 1\n").unwrap();
    let (ok, code, out, err) =
        stdout(myos(dir.path()).args(["--block-fixture", "--ref-ids", short.to_str().unwrap()]));
    assert!(ok, "stderr={err} stdout={out}");
    assert_eq!(code, 0);
    assert!(
        out.contains("CORRECT: FAIL length mismatch got=4 ref=2"),
        "{out}"
    );
}

#[test]
fn threads_flag_is_recorded_and_does_not_change_block_fixture_tokens() {
    let eight = Scratch::new("threads-8");
    let one = Scratch::new("threads-1");
    let (ok8, _, out8, err8) = stdout(myos(eight.path()).args(["--block-fixture", "--threads", "8"]));
    let (ok1, _, out1, err1) = stdout(myos(one.path()).args(["--block-fixture", "--threads", "1"]));
    assert!(ok8, "{err8}");
    assert!(ok1, "{err1}");
    assert_eq!(out8, out1);
    assert_eq!(out8, "MyOS Phase1 block fixture (prompt=0): [1, 1, 1, 1]\n");
    let json8 = std::fs::read_to_string(
        eight
            .path()
            .join("runs/s1-block-fixture/run_fingerprint.json"),
    )
    .unwrap();
    let json1 = std::fs::read_to_string(
        one.path()
            .join("runs/s1-block-fixture/run_fingerprint.json"),
    )
    .unwrap();
    assert!(json8.contains("\"threads\": 8"), "{json8}");
    assert!(json1.contains("\"threads\": 1"), "{json1}");
    assert!(json8.contains("\"gemv_pack\": false"), "{json8}");
    assert!(json8.contains("\"batch\": 1"), "{json8}");
    assert!(json8.contains("\"quant_or_dtype\": \"bf16\""), "{json8}");
    assert!(json8.contains("\"acc_path\": \"bf16_w_fp32_acc\""), "{json8}");
}

#[test]
fn arena_s1_without_weights_is_usage() {
    let dir = Scratch::new("arena-usage");
    let (ok, code, out, err) = stdout(myos(dir.path()).arg("--arena-s1"));
    assert!(!ok);
    assert_eq!(code, 2);
    assert!(err.contains("--arena-s1 needs --weights"), "{err}");
    assert_eq!(out, "");
    assert!(!dir.path().join("arena").exists());
}

#[test]
fn missing_ref_ids_does_not_print_smoke() {
    let dir = Scratch::new("missing-ref");
    let missing = dir.path().join("no-such-ref.txt");
    let (ok, code, out, err) =
        stdout(myos(dir.path()).args(["--ref-ids", missing.to_str().unwrap()]));
    assert!(!ok);
    assert_eq!(code, 1);
    assert!(err.contains("MyOS ref-ids"), "{err}");
    assert!(!out.contains("smoke decode"), "{out}");
}
