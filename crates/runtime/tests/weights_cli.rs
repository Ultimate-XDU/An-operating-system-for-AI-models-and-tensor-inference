//! Spawn `myos` and check the host weights flag without touching the cycle smoke.
use safetensors::tensor::TensorView;
use safetensors::Dtype;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn tiny_safetensors() -> Vec<u8> {
    let w = 1.0f32.to_le_bytes();
    let bias = 0.5f32.to_le_bytes();
    let w_view = TensorView::new(Dtype::F32, vec![1], &w).unwrap();
    let bias_view = TensorView::new(Dtype::F32, vec![], &bias).unwrap();
    safetensors::serialize([("w", w_view), ("bias", bias_view)], None).unwrap()
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

fn myos() -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_myos"));
    cmd.env_remove("MYOS_WEIGHTS");
    cmd
}

fn stdout(cmd: &mut Command) -> (bool, i32, String, String) {
    let output = cmd.output().unwrap();
    let code = output.status.code().unwrap_or(1);
    let out = String::from_utf8(output.stdout).unwrap();
    let err = String::from_utf8(output.stderr).unwrap();
    (output.status.success(), code, out, err)
}

#[test]
fn default_smoke_is_the_cycle() {
    let (ok, code, out, err) = stdout(&mut myos());
    assert!(ok, "stderr={err}");
    assert_eq!(code, 0);
    assert_eq!(err, "");
    assert_eq!(out, "MyOS Phase1 smoke decode (prompt=0): [1, 2, 3, 0]\n");
}

#[test]
fn weights_flag_prints_names_then_smoke() {
    let tmp = TempFile::write("flag", &tiny_safetensors());
    let (ok, code, out, err) = stdout(myos().args(["--weights", tmp.path().to_str().unwrap()]));
    assert!(ok, "stderr={err} stdout={out}");
    assert_eq!(code, 0);
    let lines: Vec<_> = out.lines().collect();
    assert_eq!(lines.len(), 4, "{out}");
    assert!(lines[0].ends_with(": 2 tensors"), "{out}");
    assert!(lines[0].contains(tmp.path().to_str().unwrap()));
    assert_eq!(lines[1], "bias");
    assert_eq!(lines[2], "w");
    assert_eq!(
        lines[3],
        "MyOS Phase1 smoke decode (prompt=0): [1, 2, 3, 0]"
    );
}

#[test]
fn env_var_prints_names_and_flag_overrides_it() {
    let good = TempFile::write("env-good", &tiny_safetensors());
    let (ok, _, out, err) = stdout(myos().env("MYOS_WEIGHTS", good.path()));
    assert!(ok, "stderr={err} stdout={out}");
    assert!(out.lines().next().unwrap().ends_with(": 2 tensors"));
    assert!(out.contains("MyOS Phase1 smoke decode (prompt=0): [1, 2, 3, 0]"));

    let missing = std::env::temp_dir().join(format!(
        "myos-runtime-missing-{}-{}.safetensors",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let (ok, _, out, err) = stdout(
        myos()
            .env("MYOS_WEIGHTS", &missing)
            .args(["--weights", good.path().to_str().unwrap()]),
    );
    assert!(ok, "flag should win over a missing env path; stderr={err}");
    assert!(out.contains("bias"));
    assert!(!out.contains(&missing.display().to_string()));
}

#[test]
fn bad_weights_do_not_print_smoke() {
    let missing = std::env::temp_dir().join(format!(
        "myos-runtime-absent-{}-{}.safetensors",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let (ok, code, out, err) = stdout(myos().args(["--weights", missing.to_str().unwrap()]));
    assert!(!ok);
    assert_eq!(code, 1);
    assert!(err.contains("MyOS weights"), "{err}");
    assert!(!out.contains("smoke decode"), "{out}");

    let garbage = TempFile::write("garbage", b"not-a-safetensors-file!!!!");
    let (ok, code, out, err) = stdout(myos().args(["--weights", garbage.path().to_str().unwrap()]));
    assert!(!ok);
    assert_eq!(code, 1);
    assert!(err.contains("could not be parsed"), "{err}");
    assert!(!out.contains("smoke decode"), "{out}");
}

#[test]
fn block_fixture_flag_is_not_the_cycle_smoke() {
    let (ok, code, out, err) = stdout(myos().arg("--block-fixture"));
    assert!(ok, "stderr={err}");
    assert_eq!(code, 0);
    assert_eq!(err, "");
    assert_eq!(out, "MyOS Phase1 block fixture (prompt=0): [1, 1, 1, 1]\n");
    assert!(!out.contains("[1, 2, 3, 0]"));
    assert!(!out.contains("smoke decode"));
}

#[test]
fn weights_then_block_fixture_skips_cycle_smoke() {
    let tmp = TempFile::write("flag-block", &tiny_safetensors());
    let (ok, _, out, err) =
        stdout(myos().args(["--weights", tmp.path().to_str().unwrap(), "--block-fixture"]));
    assert!(ok, "stderr={err} stdout={out}");
    let lines: Vec<_> = out.lines().collect();
    assert!(lines[0].ends_with(": 2 tensors"), "{out}");
    assert_eq!(lines[1], "bias");
    assert_eq!(lines[2], "w");
    assert_eq!(
        lines[3],
        "MyOS Phase1 block fixture (prompt=0): [1, 1, 1, 1]"
    );
    assert!(!out.contains("smoke decode"));
    assert!(!out.contains("[1, 2, 3, 0]"));
}

#[test]
fn bad_weights_skip_the_block_fixture() {
    let missing = std::env::temp_dir().join(format!(
        "myos-runtime-block-missing-{}-{}.safetensors",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let (ok, code, out, err) =
        stdout(myos().args(["--block-fixture", "--weights", missing.to_str().unwrap()]));
    assert!(!ok);
    assert_eq!(code, 1);
    assert!(err.contains("MyOS weights"), "{err}");
    assert!(!out.contains("block fixture"), "{out}");
    assert!(!out.contains("smoke decode"), "{out}");
}

#[test]
fn unknown_argument_is_usage() {
    let (ok, code, out, err) = stdout(myos().arg("--bogus"));
    assert!(!ok);
    assert_eq!(code, 2);
    assert!(err.contains("unknown argument"));
    assert!(err.contains("usage: myos [--weights PATH] [--block-fixture]"));
    assert_eq!(out, "");
}
