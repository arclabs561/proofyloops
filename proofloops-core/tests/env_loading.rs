use proofloops_core as plc;
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
        r#"{"mcpServers":{"proofloops":{"url":"http://127.0.0.1:8087","env":{"OPENROUTER_API_KEY":"TEST_KEY"}}}}"#,
    )
    .unwrap();

    std::env::remove_var("OPENAI_API_KEY");
    std::env::remove_var("OPENROUTER_API_KEY");
    std::env::remove_var("GROQ_API_KEY");
    std::env::set_var("PROOFLOOPS_MCP_JSON_PATH", &mcp_path);
    plc::load_cursor_mcp_env_if_present();
    assert_eq!(
        std::env::var("OPENROUTER_API_KEY").unwrap(),
        "TEST_KEY".to_string()
    );

    // cleanup
    std::env::remove_var("PROOFLOOPS_MCP_JSON_PATH");
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
    std::env::set_var("PROOFLOOPS_DOTENV_SEARCH_ROOT", td.path());
    std::env::set_var("PROOFLOOPS_DOTENV_SEARCH", "1");

    plc::load_dotenv_smart(&root);
    assert_eq!(std::env::var("OPENAI_API_KEY").unwrap(), "FROM_SIBLING");

    // cleanup for other tests
    std::env::remove_var("PROOFLOOPS_DOTENV_SEARCH_ROOT");
    std::env::remove_var("PROOFLOOPS_DOTENV_SEARCH");
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
    std::env::set_var("PROOFLOOPS_DOTENV_SEARCH_ROOT", td.path());
    std::env::set_var("PROOFLOOPS_DOTENV_SEARCH", "1");

    plc::load_dotenv_smart(&root);
    assert_eq!(std::env::var("OPENAI_API_KEY").unwrap(), "FROM_PARENT");

    // cleanup for other tests
    std::env::remove_var("PROOFLOOPS_DOTENV_SEARCH_ROOT");
    std::env::remove_var("PROOFLOOPS_DOTENV_SEARCH");
    std::env::remove_var("OPENAI_API_KEY");
}
