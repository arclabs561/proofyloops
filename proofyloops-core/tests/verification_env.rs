use proofyloops_core as plc;
use std::ffi::OsString;
use std::fs;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

fn env_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

struct EnvRestore(Vec<(&'static str, Option<OsString>)>);

impl EnvRestore {
    fn capture(names: &[&'static str]) -> Self {
        Self(
            names
                .iter()
                .map(|name| (*name, std::env::var_os(name)))
                .collect(),
        )
    }
}

impl Drop for EnvRestore {
    fn drop(&mut self) {
        for (name, value) in self.0.drain(..) {
            if let Some(value) = value {
                std::env::set_var(name, value);
            } else {
                std::env::remove_var(name);
            }
        }
    }
}

#[cfg(unix)]
#[test]
fn verification_entrypoints_do_not_load_dotenv_or_mcp_credentials() {
    use std::os::unix::fs::PermissionsExt;

    let _lock = env_lock().lock().unwrap();
    let env_names = [
        "LAKE",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "GROQ_API_KEY",
        "PROOFYLOOPS_AUTO_BUILD",
        "PROOFYLOOPS_DOTENV_SEARCH",
        "PROOFYLOOPS_MCP_JSON_PATH",
        "PROOFYLOOPS_VERIFY_BACKEND",
    ];
    let _restore = EnvRestore::capture(&env_names);

    let td = tempfile::tempdir().unwrap();
    let repo_root = td.path().join("synthetic-lean-repo");
    fs::create_dir_all(repo_root.join(".lake/build/lib/lean")).unwrap();
    fs::write(repo_root.join("lakefile.lean"), "package synthetic\n").unwrap();
    fs::write(repo_root.join("lean-toolchain"), "v4.0.0\n").unwrap();
    fs::write(
        repo_root.join("Example.lean"),
        "theorem synthetic_example : True := by\n  trivial\n",
    )
    .unwrap();
    fs::write(
        repo_root.join(".env"),
        "OPENROUTER_API_KEY=dotenv-must-not-be-loaded\n",
    )
    .unwrap();

    let mcp_path = td.path().join("mcp.json");
    fs::write(
        &mcp_path,
        r#"{"mcpServers":{"proofyloops":{"env":{"OPENAI_API_KEY":"mcp-must-not-be-loaded"}}}}"#,
    )
    .unwrap();

    let fake_lake = td.path().join("fake-lake");
    fs::write(&fake_lake, "#!/bin/sh\nexit 0\n").unwrap();
    let mut permissions = fs::metadata(&fake_lake).unwrap().permissions();
    permissions.set_mode(0o700);
    fs::set_permissions(&fake_lake, permissions).unwrap();

    for name in ["OPENAI_API_KEY", "OPENROUTER_API_KEY", "GROQ_API_KEY"] {
        std::env::remove_var(name);
    }
    std::env::set_var("LAKE", &fake_lake);
    std::env::set_var("PROOFYLOOPS_AUTO_BUILD", "0");
    std::env::set_var("PROOFYLOOPS_DOTENV_SEARCH", "0");
    std::env::set_var("PROOFYLOOPS_MCP_JSON_PATH", &mcp_path);
    std::env::set_var("PROOFYLOOPS_VERIFY_BACKEND", "lake");

    let proof_prompt = plc::build_proof_prompt(&repo_root, "Example.lean", "synthetic_example")
        .expect("proof prompt should be constructed without credential discovery");
    assert!(!proof_prompt.prompt_combined.is_empty());
    assert!(std::env::var_os("OPENROUTER_API_KEY").is_none());
    assert!(std::env::var_os("OPENAI_API_KEY").is_none());

    let rubberduck_prompt = plc::build_rubberduck_prompt(
        &repo_root,
        "Example.lean",
        "synthetic_example",
        Some("synthetic diagnostic"),
    )
    .expect("rubberduck prompt should be constructed without credential discovery");
    assert!(!rubberduck_prompt.prompt_combined.is_empty());
    assert!(rubberduck_prompt
        .user
        .contains("Lean proof synthetic_example mathlib Lean"));
    assert!(!rubberduck_prompt.user.contains("proofyloops research-auto"));
    assert!(std::env::var_os("OPENROUTER_API_KEY").is_none());
    assert!(std::env::var_os("OPENAI_API_KEY").is_none());

    let excerpt_prompt = plc::build_rubberduck_prompt_from_excerpt(
        &repo_root,
        "Example.lean",
        "synthetic_example",
        "theorem synthetic_example : True := by\n  trivial",
        None,
    )
    .expect("excerpt prompt should be constructed without credential discovery");
    assert!(!excerpt_prompt.prompt_combined.is_empty());
    assert!(std::env::var_os("OPENROUTER_API_KEY").is_none());
    assert!(std::env::var_os("OPENAI_API_KEY").is_none());

    let region_prompt = plc::build_region_patch_prompt(&repo_root, "Example.lean", 1, 2, None)
        .expect("region prompt should be constructed without credential discovery");
    assert!(!region_prompt.prompt_combined.is_empty());
    assert!(std::env::var_os("OPENROUTER_API_KEY").is_none());
    assert!(std::env::var_os("OPENAI_API_KEY").is_none());

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("test runtime");
    let text_result = runtime
        .block_on(plc::verify_lean_text(
            &repo_root,
            "#check Nat\n",
            Duration::from_secs(5),
        ))
        .expect("text verification should invoke the fake lake binary");
    assert!(text_result.ok);

    let file_result = runtime
        .block_on(plc::verify_lean_file(
            &repo_root,
            "Example.lean",
            Duration::from_secs(5),
        ))
        .expect("file verification should invoke the fake lake binary");
    assert!(file_result.ok);

    assert!(
        std::env::var_os("OPENROUTER_API_KEY").is_none(),
        "verification must not load the repository .env"
    );
    assert!(
        std::env::var_os("OPENAI_API_KEY").is_none(),
        "verification must not load MCP configuration credentials"
    );
    assert!(std::env::var_os("GROQ_API_KEY").is_none());
}
