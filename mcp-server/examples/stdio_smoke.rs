//! Smoke test for `proofpatch-mcp mcp-stdio`.
//!
//! This starts a child process running `proofpatch-mcp mcp-stdio` and calls a couple tools.
//! It validates the stdio MCP surface without relying on Cursor as the client.

#[cfg(not(feature = "stdio"))]
fn main() {
    eprintln!("stdio_smoke requires `--features stdio` (or default features enabled)");
}

#[cfg(feature = "stdio")]
use rmcp::{
    model::CallToolRequestParam,
    service::ServiceExt,
    transport::{ConfigureCommandExt, TokioChildProcess},
};
#[cfg(feature = "stdio")]
// keep serde_json in scope for json! macro usage
use serde_json as _;
#[cfg(feature = "stdio")]
use std::path::PathBuf;
#[cfg(feature = "stdio")]
use tokio::process::Command;

#[cfg(feature = "stdio")]
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // `CARGO_MANIFEST_DIR` here is `.../proofpatch/mcp-server`.
    // The binary is built into the *workspace* `target/` by default.
    let workspace_root = root
        .parent()
        .expect("mcp-server should be nested under proofpatch/")
        .to_path_buf();
    let bin = workspace_root.join("target/debug/proofpatch-mcp");
    if !bin.exists() {
        anyhow::bail!(
            "missing server binary at {}\n\nBuild it with:\n  cargo build -p proofpatch-mcp --bin proofpatch-mcp --features stdio",
            bin.display()
        );
    }
    eprintln!("spawning: {} mcp-stdio", bin.display());

    // Make the smoke test independent of any particular repo layout.
    //
    // This is a true smoke test of the stdio MCP surface. Require an explicit Lean repo root so
    // we don't bake in developer-specific paths in a public repo.
    let repo_root = std::env::var("PROOFPATCH_SMOKE_REPO_ROOT").map_err(|_| {
        anyhow::anyhow!(
            "PROOFPATCH_SMOKE_REPO_ROOT is required (set it to an absolute Lean repo path)"
        )
    })?;
    let file = std::env::var("PROOFPATCH_SMOKE_FILE")
        .unwrap_or_else(|_| "GeometryOfNumbers/Legendre/Main.lean".to_string());

    let service = ()
        .serve(TokioChildProcess::new(Command::new(&bin).configure(
            |cmd| {
                cmd.arg("mcp-stdio");
            },
        ))?)
        .await?;

    let info = service.peer_info();
    println!("peer_info: {:#?}", info);

    let tools = service.list_tools(Default::default()).await?;
    println!("tools: {:#?}", tools);

    // Default toolset is "minimal": it exposes a single `proofpatch` tool that dispatches on
    // `{ action, arguments }`. Keep the smoke test compatible with both minimal and full toolsets.
    //
    // Keep this cheap: triage one file with a small timeout.
    let triage = service
        .call_tool(CallToolRequestParam {
            name: "proofpatch".into(),
            arguments: Some(
                serde_json::json!({
                    "action": "triage_file",
                    "arguments": {
                        "repo_root": repo_root.clone(),
                        "file": file.clone(),
                        "timeout_s": 120,
                        "max_sorries": 3,
                        "context_lines": 1
                    }
                })
                .as_object()
                .cloned()
                .unwrap_or_default(),
            ),
        })
        .await?;
    println!("proofpatch (triage_file): {:#?}", triage);

    let pack = service
        .call_tool(CallToolRequestParam {
            name: "proofpatch".into(),
            arguments: Some(
                serde_json::json!({
                    "action": "context_pack",
                    "arguments": {
                        "repo_root": repo_root.clone(),
                        "file": file.clone(),
                        // Fixture decl name (keep this independent of any external repo).
                        "decl": "one_plus_one_eq_two",
                        "context_lines": 20,
                        "nearby_lines": 60,
                        "max_nearby_decls": 20,
                        "max_imports": 20
                    }
                })
                .as_object()
                .cloned()
                .unwrap_or_default(),
            ),
        })
        .await?;
    println!("proofpatch (context_pack): {:#?}", pack);

    // Exercise another action: locate sorries (fixture should usually be sorry-free).
    let locate = service
        .call_tool(CallToolRequestParam {
            name: "proofpatch".into(),
            arguments: Some(
                serde_json::json!({
                    "action": "locate_sorries",
                    "arguments": {
                        "repo_root": repo_root.clone(),
                        "file": file.clone(),
                        "max_results": 10,
                        "context_lines": 2
                    }
                })
                .as_object()
                .cloned()
                .unwrap_or_default(),
            ),
        })
        .await?;
    println!("proofpatch (locate_sorries): {:#?}", locate);

    service.cancel().await?;
    Ok(())
}
