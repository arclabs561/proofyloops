//! Smoke test for `proofyloops-mcp mcp-stdio`.
//!
//! This starts a child process running `proofyloops-mcp mcp-stdio` and calls a couple tools.
//! It validates the stdio MCP surface without relying on Cursor as the client.

use rmcp::{
    model::{CallToolRequestParams, CallToolResult},
    service::ServiceExt,
    transport::{ConfigureCommandExt, TokioChildProcess},
};
// keep serde_json in scope for json! macro usage
use serde_json as _;
use std::path::PathBuf;
use tokio::process::Command;

#[derive(Clone, Copy)]
enum Toolset {
    Minimal,
    Full,
}

fn call_params(
    toolset: Toolset,
    action: &str,
    arguments: serde_json::Value,
) -> CallToolRequestParams {
    let arguments = arguments
        .as_object()
        .cloned()
        .expect("smoke test arguments must be JSON objects");
    match toolset {
        Toolset::Minimal => CallToolRequestParams::new("proofyloops").with_arguments(
            serde_json::json!({ "action": action, "arguments": arguments })
                .as_object()
                .cloned()
                .expect("minimal MCP request must be a JSON object"),
        ),
        Toolset::Full => {
            let tool = match action {
                "triage_file" => "proofyloops_triage_file",
                "context_pack" => "proofyloops_context_pack",
                "locate_sorries" => "proofyloops_locate_sorries",
                _ => panic!("unsupported full-toolset smoke action: {action}"),
            };
            CallToolRequestParams::new(tool).with_arguments(arguments)
        }
    }
}

fn result_json(label: &str, result: CallToolResult) -> anyhow::Result<serde_json::Value> {
    anyhow::ensure!(
        result.is_error != Some(true),
        "{label} returned an MCP tool error: {result:#?}"
    );
    let text = result
        .content
        .iter()
        .find_map(|content| content.as_text())
        .ok_or_else(|| anyhow::anyhow!("{label} returned no text content: {result:#?}"))?;
    serde_json::from_str(&text.text).map_err(|error| {
        anyhow::anyhow!(
            "{label} returned non-JSON text: {error}; text={}",
            text.text
        )
    })
}

fn assert_sorry_free_triage(value: &serde_json::Value) -> anyhow::Result<()> {
    let summary = &value["verify"]["summary"];
    anyhow::ensure!(
        summary["ok"] == true,
        "triage verification failed: {value:#}"
    );
    anyhow::ensure!(
        summary["timeout"] == false,
        "triage verification timed out: {value:#}"
    );
    anyhow::ensure!(
        summary["returncode"] == 0,
        "triage Lean exit code was not zero: {value:#}"
    );
    anyhow::ensure!(
        summary["counts"]["errors"] == 0,
        "triage reported errors: {value:#}"
    );
    anyhow::ensure!(
        summary["counts"]["sorry_warnings"] == 0,
        "triage reported admitted declarations: {value:#}"
    );
    anyhow::ensure!(
        value["sorries"]["count"] == 0,
        "triage located holes: {value:#}"
    );
    if let Some(count) = value["sorries"]["conservative_count"].as_u64() {
        anyhow::ensure!(count == 0, "triage conservatively located holes: {value:#}");
    }
    anyhow::ensure!(
        value["nearest_sorry_to_first_error"].is_null(),
        "triage selected a hole: {value:#}"
    );
    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // `CARGO_MANIFEST_DIR` here is `.../proofyloops/mcp-server`.
    // The binary is built into the *workspace* `target/` by default.
    let workspace_root = root
        .parent()
        .expect("mcp-server should be nested under proofyloops/")
        .to_path_buf();
    let bin = workspace_root.join("target/debug/proofyloops-mcp");
    if !bin.exists() {
        anyhow::bail!(
            "missing server binary at {}\n\nBuild it with:\n  cargo build -p proofyloops-mcp --bin proofyloops-mcp",
            bin.display()
        );
    }
    eprintln!("spawning: {} mcp-stdio", bin.display());

    // Make the smoke test independent of any particular repo layout.
    //
    // This is a true smoke test of the stdio MCP surface. Require an explicit Lean repo root so
    // we don't bake in developer-specific paths in a public repo.
    let repo_root = std::env::var("PROOFYLOOPS_SMOKE_REPO_ROOT").map_err(|_| {
        anyhow::anyhow!(
            "PROOFYLOOPS_SMOKE_REPO_ROOT is required (set it to an absolute Lean repo path)"
        )
    })?;
    let file = std::env::var("PROOFYLOOPS_SMOKE_FILE").map_err(|_| {
        anyhow::anyhow!(
            "PROOFYLOOPS_SMOKE_FILE is required (set it to a file relative to the fixture repo)"
        )
    })?;
    let decl = std::env::var("PROOFYLOOPS_SMOKE_DECL").map_err(|_| {
        anyhow::anyhow!(
            "PROOFYLOOPS_SMOKE_DECL is required (set it to a declaration in PROOFYLOOPS_SMOKE_FILE)"
        )
    })?;

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
    let toolset = if tools.tools.iter().any(|tool| tool.name == "proofyloops") {
        Toolset::Minimal
    } else {
        for required in [
            "proofyloops_triage_file",
            "proofyloops_context_pack",
            "proofyloops_locate_sorries",
        ] {
            anyhow::ensure!(
                tools.tools.iter().any(|tool| tool.name == required),
                "MCP server exposed neither the minimal tool nor required full tool {required}"
            );
        }
        Toolset::Full
    };

    // Default toolset is "minimal": it exposes a single `proofyloops` tool that dispatches on
    // `{ action, arguments }`. Keep the smoke test compatible with both minimal and full toolsets.
    //
    // Keep this cheap: triage one file with a small timeout.
    let triage = service
        .call_tool(call_params(
            toolset,
            "triage_file",
            serde_json::json!({
                "repo_root": repo_root.clone(),
                "file": file.clone(),
                "timeout_s": 120,
                "max_sorries": 3,
                "context_lines": 1
            }),
        ))
        .await?;
    let triage = result_json("triage_file", triage)?;
    assert_sorry_free_triage(&triage)?;
    println!("triage_file: {triage:#}");

    let pack = service
        .call_tool(call_params(
            toolset,
            "context_pack",
            serde_json::json!({
                "repo_root": repo_root.clone(),
                "file": file.clone(),
                // Fixture decl name is supplied by CI so this remains fixture-specific.
                "decl": decl.clone(),
                "context_lines": 20,
                "nearby_lines": 60,
                "max_nearby_decls": 20,
                "max_imports": 20
            }),
        ))
        .await?;
    let pack = result_json("context_pack", pack)?;
    anyhow::ensure!(
        pack["file_rel"] == file,
        "context pack used the wrong file: {pack:#}"
    );
    anyhow::ensure!(
        pack["focus"]["decl"] == decl,
        "context pack did not resolve the requested declaration: {pack:#}"
    );
    anyhow::ensure!(
        pack["focus"]["excerpt"]
            .as_str()
            .is_some_and(|text| !text.is_empty()),
        "context pack has no excerpt: {pack:#}"
    );
    println!("context_pack: {pack:#}");

    // Exercise another action: locate sorries (fixture should usually be sorry-free).
    let locate = service
        .call_tool(call_params(
            toolset,
            "locate_sorries",
            serde_json::json!({
                "repo_root": repo_root.clone(),
                "file": file.clone(),
                "max_results": 10,
                "context_lines": 2
            }),
        ))
        .await?;
    let locate = result_json("locate_sorries", locate)?;
    anyhow::ensure!(
        locate["count"] == 0,
        "locate_sorries found holes: {locate:#}"
    );
    println!("locate_sorries: {locate:#}");

    // A valid MCP response with a nonexistent file must be rejected, proving this smoke test
    // does not mistake transport success for a successful tool outcome.
    let missing = service
        .call_tool(call_params(
            toolset,
            "locate_sorries",
            serde_json::json!({
                "repo_root": repo_root,
                "file": "MissingFixtureFile.lean",
                "max_results": 1,
                "context_lines": 1
            }),
        ))
        .await;
    anyhow::ensure!(
        missing.is_err(),
        "missing-file tool call unexpectedly succeeded"
    );

    service.cancel().await?;
    Ok(())
}
