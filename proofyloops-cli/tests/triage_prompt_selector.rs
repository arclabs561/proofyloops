use std::fs;
use std::process::Command;

#[cfg(unix)]
#[test]
fn triage_prompt_uses_the_repo_owned_declaration_selector() {
    use std::os::unix::fs::PermissionsExt;

    let temp = tempfile::tempdir().expect("tempdir");
    let repo = temp.path().join("lean-repo");
    fs::create_dir_all(repo.join(".lake/build/lib/lean")).expect("lean build dir");
    fs::write(repo.join("lakefile.lean"), "package demo\n").expect("lakefile");
    fs::write(repo.join("lean-toolchain"), "v4.0.0\n").expect("toolchain");
    fs::write(
        repo.join("Example.lean"),
        "theorem selected_decl : True := by\n  sorry\n",
    )
    .expect("source");
    fs::write(
        repo.join("proofyloops.toml"),
        "[research.presets.selected]\nquery = \"custom report query\"\nwhen_decl_contains = [\"selected_decl\"]\n",
    )
    .expect("config");

    let lake = temp.path().join("fake-lake");
    fs::write(
        &lake,
        "#!/bin/sh\nprintf '%s:1:1: error: synthetic failure\\n' \"$4\" >&2\nexit 1\n",
    )
    .expect("fake lake");
    let mut permissions = fs::metadata(&lake).expect("lake metadata").permissions();
    permissions.set_mode(0o700);
    fs::set_permissions(&lake, permissions).expect("lake executable");

    let output = Command::new(env!("CARGO_BIN_EXE_proofyloops"))
        .args([
            "triage-file",
            "--repo",
            repo.to_str().expect("repo path"),
            "--file",
            "Example.lean",
            "--timeout-s",
            "5",
        ])
        .env("LAKE", lake)
        .env("PROOFYLOOPS_AUTO_BUILD", "0")
        .env("PROOFYLOOPS_VERIFY_BACKEND", "lake")
        .output()
        .expect("run triage-file");

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("triage JSON report");
    let prompt = report
        .pointer("/rubberduck_prompt_first_error/user")
        .and_then(serde_json::Value::as_str)
        .expect("first-error prompt");
    assert!(prompt.contains("custom report query mathlib Lean"));
    assert!(prompt.contains("\"preset\":\"selected\""));
    assert!(!prompt.contains("\"preset\":\"<preset_name>\""));
}
