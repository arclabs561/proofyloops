#![cfg(unix)]

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::process::Command;

// Fake only the external verifier process; run the actual CLI and JSON routing.
fn triage(verifier: &str, timeout_s: &str) -> serde_json::Value {
    let dir = tempfile::tempdir().unwrap();
    fs::create_dir_all(dir.path().join(".lake/build/lib/lean")).unwrap();
    fs::write(dir.path().join("lakefile.lean"), "package fixture\n").unwrap();
    fs::write(dir.path().join("lean-toolchain"), "v4.0.0\n").unwrap();
    fs::write(
        dir.path().join("Complete.lean"),
        "theorem complete_example : True := by trivial\n",
    )
    .unwrap();
    let lake = dir.path().join("fake-lake");
    fs::write(&lake, format!("#!/bin/sh\n{verifier}\n")).unwrap();
    fs::set_permissions(&lake, fs::Permissions::from_mode(0o700)).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_proofyloops"))
        .args(["triage-file", "--repo"])
        .arg(dir.path())
        .args([
            "--file",
            "Complete.lean",
            "--timeout-s",
            timeout_s,
            "--no-prompts",
            "--no-context-pack",
        ])
        .env_clear()
        .env("LAKE", lake)
        .env("PROOFYLOOPS_AUTO_BUILD", "0")
        .env("PROOFYLOOPS_VERIFY_BACKEND", "lake")
        .output()
        .unwrap();
    assert!(output.status.success(), "{output:?}");
    serde_json::from_slice(&output.stdout).unwrap()
}

#[test]
fn interrupted_verification_does_not_recommend_noop() {
    let result = triage("exec /bin/sleep 1", "0");
    assert_eq!(result["verify"]["summary"]["timeout"], true);
    assert_eq!(result["next_action"]["kind"], "retry_verification");
    assert_eq!(result["next_action"]["reason"], "timeout");
}

#[test]
fn unparsed_process_failure_does_not_recommend_noop() {
    let result = triage("exit 9", "5");
    assert_eq!(result["verify"]["summary"]["ok"], false);
    assert_eq!(result["verify"]["summary"]["counts"]["errors"], 0);
    assert_eq!(
        result["next_action"]["kind"],
        "inspect_verification_failure"
    );
}

#[test]
fn successful_complete_verification_still_recommends_noop() {
    let result = triage("exit 0", "5");
    assert_eq!(result["verify"]["summary"]["ok"], true);
    assert_eq!(result["next_action"]["kind"], "noop");
}
