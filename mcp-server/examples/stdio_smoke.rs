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

    // Keep this cheap: triage one file with a small timeout.
    let triage = service
        .call_tool(CallToolRequestParam {
            name: "proofpatch_triage_file".into(),
            arguments: Some(
                serde_json::json!({
                    "repo_root": repo_root.clone(),
                    "file": file.clone(),
                    "timeout_s": 120,
                    "max_sorries": 3,
                    "context_lines": 1
                })
                .as_object()
                .cloned()
                .unwrap_or_default(),
            ),
        })
        .await?;
    println!("proofpatch_triage_file: {:#?}", triage);

    let pack = service
        .call_tool(CallToolRequestParam {
            name: "proofpatch_context_pack".into(),
            arguments: Some(
                serde_json::json!({
                    "repo_root": repo_root.clone(),
                    "file": file.clone(),
                    "decl": "cauchy_decomposition",
                    "context_lines": 20,
                    "nearby_lines": 60,
                    "max_nearby_decls": 20,
                    "max_imports": 20
                })
                .as_object()
                .cloned()
                .unwrap_or_default(),
            ),
        })
        .await?;
    println!("proofpatch_context_pack: {:#?}", pack);

    // Exercise the expanded stdio surface (typed schemas).
    let locate = service
        .call_tool(CallToolRequestParam {
            name: "proofpatch_locate_sorries".into(),
            arguments: Some(
                serde_json::json!({
                    "repo_root": repo_root.clone(),
                    "file": "GeometryOfNumbers/Legendre/Main.lean",
                    "max_results": 10,
                    "context_lines": 2
                })
                .as_object()
                .cloned()
                .unwrap_or_default(),
            ),
        })
        .await?;
    println!("proofpatch_locate_sorries (Legendre/Main): {:#?}", locate);

    let rubberduck = service
        .call_tool(CallToolRequestParam {
            name: "proofpatch_rubberduck_prompt".into(),
            arguments: Some(
                serde_json::json!({
                    "repo_root": repo_root.clone(),
                    "file": "Covolume/Legendre/Main.lean",
                    "lemma": "sum_three_squares_of_not_exception"
                })
                .as_object()
                .cloned()
                .unwrap_or_default(),
            ),
        })
        .await?;
    println!(
        "proofpatch_rubberduck_prompt (Legendre/Main.sum_three_squares_of_not_exception): {:#?}",
        rubberduck
    );

    let step = service
        .call_tool(CallToolRequestParam {
            name: "proofpatch_agent_step".into(),
            arguments: Some(
                serde_json::json!({
                    "repo_root": repo_root.clone(),
                    "file": file.clone(),
                    "timeout_s": 120,
                    "write": false
                })
                .as_object()
                .cloned()
                .unwrap_or_default(),
            ),
        })
        .await?;
    println!("proofpatch_agent_step: {:#?}", step);

    service.cancel().await?;
    Ok(())
}
