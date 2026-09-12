use std::process::Command;

#[cfg(not(feature = "lean-embed"))]
#[test]
fn disabled_embedding_names_the_cli_package_in_recovery_command() {
    let output = Command::new(env!("CARGO_BIN_EXE_proofyloops"))
        .arg("lean-embed-smoke")
        .env_clear()
        .output()
        .unwrap();
    assert!(!output.status.success());
    let error = String::from_utf8(output.stderr).unwrap();
    assert!(error.contains("cargo run -p proofyloops --features lean-embed --bin proofyloops"));
}

#[cfg(feature = "lean-embed")]
#[test]
fn embedded_lean_command_returns_the_computed_sum() {
    let output = Command::new(env!("CARGO_BIN_EXE_proofyloops"))
        .arg("lean-embed-smoke")
        .current_dir(std::env::temp_dir())
        .env_clear()
        .output()
        .unwrap();
    assert!(output.status.success(), "{output:?}");
    let result: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(result["ok"], true);
    assert_eq!(result["result"], 42);
}

#[test]
fn version_is_available_without_a_repo_or_credentials() {
    for flag in ["--version", "-V"] {
        let output = Command::new(env!("CARGO_BIN_EXE_proofyloops"))
            .arg(flag)
            .current_dir(std::env::temp_dir())
            .env_clear()
            .output()
            .unwrap();
        assert!(output.status.success(), "{output:?}");
        assert_eq!(
            String::from_utf8(output.stdout).unwrap(),
            format!("proofyloops {}\n", env!("CARGO_PKG_VERSION"))
        );
        assert!(output.stderr.is_empty());
    }
}
