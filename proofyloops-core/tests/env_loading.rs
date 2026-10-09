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
