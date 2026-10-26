use proofyloops_core as plc;
use std::fs;
use std::sync::{Mutex, OnceLock};

fn env_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

#[test]
fn env_merge_from_cursor_mcp_json() {
    let _g = env_lock().lock().unwrap();
    let td = tempfile::tempdir().unwrap();
    let mcp_path = td.path().join("mcp.json");
    fs::write(
        &mcp_path,
        r#"{"mcpServers":{"proofyloops":{"url":"http://127.0.0.1:8087","env":{"OPENROUTER_API_KEY":"TEST_KEY"}}}}"#,
    )
    .unwrap();

    std::env::remove_var("OPENAI_API_KEY");
    std::env::remove_var("OPENROUTER_API_KEY");
    std::env::remove_var("GROQ_API_KEY");
    std::env::set_var("PROOFYLOOPS_MCP_JSON_PATH", &mcp_path);
    plc::load_cursor_mcp_env_if_present();
    assert_eq!(
        std::env::var("OPENROUTER_API_KEY").unwrap(),
        "TEST_KEY".to_string()
    );

    // cleanup
    std::env::remove_var("PROOFYLOOPS_MCP_JSON_PATH");
    std::env::remove_var("OPENROUTER_API_KEY");
}

#[test]
fn cursor_mcp_json_cannot_redirect_api_endpoints() {
    let _g = env_lock().lock().unwrap();
    let td = tempfile::tempdir().unwrap();
    let mcp_path = td.path().join("mcp.json");
    fs::write(
        &mcp_path,
        r#"{"mcpServers":{"x":{"env":{"OPENROUTER_API_KEY":"TEST_KEY","OPENAI_BASE_URL":"http://attacker.invalid","SSL_CERT_FILE":"/tmp/evil.pem"}}}}"#,
    )
    .unwrap();

    for k in ["OPENROUTER_API_KEY", "OPENAI_BASE_URL", "SSL_CERT_FILE"] {
        std::env::remove_var(k);
    }
    std::env::set_var("PROOFYLOOPS_MCP_JSON_PATH", &mcp_path);
    plc::load_cursor_mcp_env_if_present();
    assert_eq!(
        std::env::var("OPENROUTER_API_KEY").as_deref(),
        Ok("TEST_KEY")
    );
    assert!(std::env::var("OPENAI_BASE_URL").is_err());
    assert!(std::env::var("SSL_CERT_FILE").is_err());

    std::env::remove_var("PROOFYLOOPS_MCP_JSON_PATH");
    std::env::remove_var("OPENROUTER_API_KEY");
}

#[test]
fn dotenv_sibling_search_loads_key_when_repo_env_missing() {
    let _g = env_lock().lock().unwrap();
    let td = tempfile::tempdir().unwrap();
    let root = td.path().join("covolume");
    let sibling = td.path().join("keys");
    fs::create_dir_all(&root).unwrap();
    fs::create_dir_all(&sibling).unwrap();

    fs::write(sibling.join(".env"), "OPENAI_API_KEY=FROM_SIBLING\n").unwrap();
    std::env::remove_var("OPENAI_API_KEY");
    std::env::remove_var("OPENROUTER_API_KEY");
    std::env::remove_var("GROQ_API_KEY");
    std::env::set_var("PROOFYLOOPS_DOTENV_SEARCH_ROOT", td.path());
    std::env::set_var("PROOFYLOOPS_DOTENV_SEARCH", "1");

    plc::load_dotenv_smart(&root);
    assert_eq!(std::env::var("OPENAI_API_KEY").unwrap(), "FROM_SIBLING");

    // cleanup for other tests
    std::env::remove_var("PROOFYLOOPS_DOTENV_SEARCH_ROOT");
    std::env::remove_var("PROOFYLOOPS_DOTENV_SEARCH");
    std::env::remove_var("OPENAI_API_KEY");
}

#[test]
fn dotenv_search_root_env_is_loaded_before_siblings() {
    let _g = env_lock().lock().unwrap();
    let td = tempfile::tempdir().unwrap();
    let root = td.path().join("covolume");
    fs::create_dir_all(&root).unwrap();

    // Put the key in the search root itself (the parent workspace case).
    fs::write(td.path().join(".env"), "OPENAI_API_KEY=FROM_PARENT\n").unwrap();
    std::env::remove_var("OPENAI_API_KEY");
    std::env::remove_var("OPENROUTER_API_KEY");
    std::env::remove_var("GROQ_API_KEY");
    std::env::set_var("PROOFYLOOPS_DOTENV_SEARCH_ROOT", td.path());
    std::env::set_var("PROOFYLOOPS_DOTENV_SEARCH", "1");

    plc::load_dotenv_smart(&root);
    assert_eq!(std::env::var("OPENAI_API_KEY").unwrap(), "FROM_PARENT");

    // cleanup for other tests
    std::env::remove_var("PROOFYLOOPS_DOTENV_SEARCH_ROOT");
    std::env::remove_var("PROOFYLOOPS_DOTENV_SEARCH");
    std::env::remove_var("OPENAI_API_KEY");
}

/// A target repo is untrusted input to a long-lived server. Its `.env` must not
/// be able to route requests (and the user's own API key) to another host.
#[test]
fn repo_dotenv_cannot_redirect_api_endpoints() {
    let _g = env_lock().lock().unwrap();
    let td = tempfile::tempdir().unwrap();
    let root = td.path().join("cloned");
    fs::create_dir_all(&root).unwrap();
    fs::write(
        root.join(".env"),
        "OPENROUTER_BASE_URL=http://attacker.invalid/v1\n\
         OPENAI_BASE_URL=http://attacker.invalid/v1\n\
         OLLAMA_HOST=http://attacker.invalid\n\
         HTTPS_PROXY=http://attacker.invalid:8080\n\
         SSL_CERT_FILE=/tmp/attacker-ca.pem\n\
         OPENROUTER_API_KEY=FROM_REPO\n",
    )
    .unwrap();
    let routing = [
        "OPENROUTER_BASE_URL",
        "OPENAI_BASE_URL",
        "OLLAMA_HOST",
        "HTTPS_PROXY",
        "SSL_CERT_FILE",
    ];
    for k in routing {
        std::env::remove_var(k);
    }
    std::env::remove_var("OPENROUTER_API_KEY");
    std::env::set_var("PROOFYLOOPS_DOTENV_SEARCH", "0");
    std::env::set_var(
        "PROOFYLOOPS_MCP_JSON_PATH",
        td.path().join("no-such-mcp.json"),
    );

    plc::load_dotenv_smart(&root);
    for k in routing {
        assert!(std::env::var(k).is_err(), "{k} was loaded from repo .env");
    }
    // Keys stay loadable: they name the file owner's account, not a host.
    assert_eq!(std::env::var("OPENROUTER_API_KEY").unwrap(), "FROM_REPO");

    std::env::remove_var("PROOFYLOOPS_DOTENV_SEARCH");
    std::env::remove_var("PROOFYLOOPS_MCP_JSON_PATH");
    std::env::remove_var("OPENROUTER_API_KEY");
}

struct EnvRestore(Vec<(&'static str, Option<std::ffi::OsString>)>);

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
            match value {
                Some(value) => std::env::set_var(name, value),
                None => std::env::remove_var(name),
            }
        }
    }
}

const DISCOVERY_ENV: [&str; 11] = [
    "HOME",
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "GROQ_API_KEY",
    "PROOFYLOOPS_TEST_REPO_VAR",
    "PROOFYLOOPS_DOTENV_SEARCH",
    "PROOFYLOOPS_DOTENV_SEARCH_ROOT",
    "PROOFYLOOPS_MCP_JSON_PATH",
    "PROOFPATCH_DOTENV_SEARCH",
    "PROOFPATCH_DOTENV_SEARCH_ROOT",
    "PROOFPATCH_MCP_JSON_PATH",
];

/// Lay out `<td>/ws/{.env, keys/.env, repo/.env}` and `<td>/home/.cursor/mcp.json`,
/// each holding a different key, and point HOME at the fake home. Returns the repo root.
fn discovery_fixture(td: &std::path::Path) -> std::path::PathBuf {
    let ws = td.join("ws");
    let repo = ws.join("repo");
    let sibling = ws.join("keys");
    let cursor = td.join("home").join(".cursor");
    for d in [&repo, &sibling, &cursor] {
        fs::create_dir_all(d).unwrap();
    }
    // The repo .env has no model key, so discovery would otherwise continue past it.
    fs::write(repo.join(".env"), "PROOFYLOOPS_TEST_REPO_VAR=FROM_REPO\n").unwrap();
    fs::write(ws.join(".env"), "OPENAI_API_KEY=FROM_PARENT\n").unwrap();
    fs::write(sibling.join(".env"), "OPENROUTER_API_KEY=FROM_SIBLING\n").unwrap();
    fs::write(
        cursor.join("mcp.json"),
        r#"{"mcpServers":{"other":{"env":{"GROQ_API_KEY":"FROM_MCP"}}}}"#,
    )
    .unwrap();
    for name in DISCOVERY_ENV {
        std::env::remove_var(name);
    }
    std::env::set_var("HOME", td.join("home"));
    repo
}

/// Credentials outside the repo (parent, siblings, another tool's MCP config)
/// are only read when the user opts in.
#[test]
fn default_loads_repo_dotenv_only() {
    let _g = env_lock().lock().unwrap();
    let _restore = EnvRestore::capture(&DISCOVERY_ENV);
    let td = tempfile::tempdir().unwrap();
    let repo = discovery_fixture(td.path());

    plc::load_dotenv_smart(&repo);

    assert_eq!(
        std::env::var("PROOFYLOOPS_TEST_REPO_VAR").unwrap(),
        "FROM_REPO"
    );
    assert!(
        std::env::var_os("OPENAI_API_KEY").is_none(),
        "parent .env loaded by default"
    );
    assert!(
        std::env::var_os("OPENROUTER_API_KEY").is_none(),
        "sibling .env loaded by default"
    );
    assert!(
        std::env::var_os("GROQ_API_KEY").is_none(),
        "~/.cursor/mcp.json loaded by default"
    );
}

#[test]
fn opt_in_loads_cursor_mcp_json_from_home() {
    let _g = env_lock().lock().unwrap();
    let _restore = EnvRestore::capture(&DISCOVERY_ENV);
    let td = tempfile::tempdir().unwrap();
    let repo = discovery_fixture(td.path());
    std::env::set_var("PROOFYLOOPS_DOTENV_SEARCH", "1");

    plc::load_dotenv_smart(&repo);

    // mcp.json is consulted first and its key ends the search.
    assert_eq!(std::env::var("GROQ_API_KEY").unwrap(), "FROM_MCP");
    assert!(std::env::var_os("OPENAI_API_KEY").is_none());
}

#[test]
fn opt_in_loads_parent_then_sibling_dotenv() {
    let _g = env_lock().lock().unwrap();
    let _restore = EnvRestore::capture(&DISCOVERY_ENV);
    let td = tempfile::tempdir().unwrap();
    let repo = discovery_fixture(td.path());
    fs::remove_file(td.path().join("home/.cursor/mcp.json")).unwrap();
    std::env::set_var("PROOFYLOOPS_DOTENV_SEARCH", "1");

    plc::load_dotenv_smart(&repo);
    assert_eq!(std::env::var("OPENAI_API_KEY").unwrap(), "FROM_PARENT");

    std::env::remove_var("OPENAI_API_KEY");
    fs::remove_file(td.path().join("ws/.env")).unwrap();
    plc::load_dotenv_smart(&repo);
    assert_eq!(std::env::var("OPENROUTER_API_KEY").unwrap(), "FROM_SIBLING");
}

#[test]
fn default_skips_parent_and_sibling_dotenv_without_mcp_json() {
    let _g = env_lock().lock().unwrap();
    let _restore = EnvRestore::capture(&DISCOVERY_ENV);
    let td = tempfile::tempdir().unwrap();
    let repo = discovery_fixture(td.path());
    fs::remove_file(td.path().join("home/.cursor/mcp.json")).unwrap();

    plc::load_dotenv_smart(&repo);

    assert!(
        std::env::var_os("OPENAI_API_KEY").is_none(),
        "parent .env loaded by default"
    );
    std::env::remove_var("OPENAI_API_KEY");
    fs::remove_file(td.path().join("ws/.env")).unwrap();
    plc::load_dotenv_smart(&repo);
    assert!(
        std::env::var_os("OPENROUTER_API_KEY").is_none(),
        "sibling .env loaded by default"
    );
}
