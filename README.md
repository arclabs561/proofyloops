# proofyloops

CLI and MCP server for debuggable Lean 4 proof workflows.

`proofyloops` works on files and declarations in any Lean 4 project: verify
them, locate their `sorry`s, extract bounded context packs, and patch them.
Optional commands call an OpenAI-compatible model. For tactic-level proof
search or dataset extraction, use Pantograph or LeanDojo; for an MCP view of
Lean's language server (goals, diagnostics, hovers), use lean-lsp-mcp.

Commands take `--repo <lean-project>` plus a target file, print JSON to stdout,
and write artifacts under `.generated/` when asked.

## Quickstart (CLI)

From this repo, triage the bundled fixture (needs `lake` on PATH):

```bash
cargo run -p proofyloops --bin proofyloops -- triage-file \
  --repo fixtures/lean-fixture \
  --file LeanFixture/SmtDemo.lean \
  | jq '{ok: .verify.summary.ok, counts: .verify.summary.counts,
         sorries: [.sorries.locations[] | {line, decl_name}],
         next: .next_action.kind}'
```

```json
{
  "ok": true,
  "counts": { "errors": 0, "sorry_warnings": 3, "warnings": 3 },
  "sorries": [
    { "line": 12, "decl_name": "smt_int_trivial" },
    { "line": 15, "decl_name": "smt_nat_monotone" },
    { "line": 18, "decl_name": "smt_int_monotone" }
  ],
  "next": "patch_nearest_sorry"
}
```

`ok: true` means the file compiles; the three `sorry` warnings mean it is not
complete. Run `proofyloops --help` for the full command list.

No MCP client or model account is needed for local triage, verification, or context
packs. To install the CLI from this checkout:

```bash
cargo install --path proofyloops-cli
```

See [CLI usage](docs/usage.md) for checking proof completeness, controlling builds,
and interpreting JSON results.

## Migrating from Proofpatch

The CLI is now `proofyloops`; the server is `proofyloops-mcp`. Reinstall from
this checkout and update executable paths in your MCP client, then restart it
to discover the renamed `proofyloops` and `proofyloops_*` tools.

Use `proofyloops.toml` and `PROOFYLOOPS_*` settings. The old `proofpatch.toml`
filename and `PROOFPATCH_*` runtime environment variables remain fallbacks;
the new names take precedence. Rust package names and the Lean helper import
are now `proofyloops-core` and `ProofyloopsTools`, respectively.

## MCP server

The server speaks MCP over stdio:

```bash
cargo run -p proofyloops-mcp --bin proofyloops-mcp
```

## SMT (optional)

`proofyloops` can use [`smtkit`](https://crates.io/crates/smtkit) to check
linear integer arithmetic entailments. The result is a heuristic signal for
ranking candidates, never a proof.

With an SMT solver such as `z3` on PATH:

```bash
proofyloops smt-probe

proofyloops tree-search-nearest \
  --repo /abs/path/to/lean-repo \
  --file Some/File.lean \
  --goal-dump \
  --smt-precheck \
  --smt-dump --smt-dump-dir .generated/proofyloops-smt2 \
  --smt-repro-dir .generated/proofyloops-smtrepro \
  --output-json .generated/proofyloops-tree-search.json
```

- `--goal-dump` extracts hypotheses and targets, which `--smt-precheck` needs
  for informative checks.
- `--smt-dump` writes bounded `.smt2` scripts to disk for inspection.
- `--smt-repro-dir` writes a repro bundle; re-run it with
  `proofyloops smt-repro --input-json .generated/proofyloops-smtrepro/tree_search.json`.

[docs/smt.md](docs/smt.md) covers dump and repro bundles, UNSAT core and proof
capture options, and `smt-repro`.

## More docs

- [docs/usage.md](docs/usage.md): common CLI patterns, focus flags, output stability.
- [lean-tools/README.md](lean-tools/README.md): `ProofyloopsTools` (Lean-side helper tactics).
- [proofyloops-lean-embed/README.md](proofyloops-lean-embed/README.md): optional Lean runtime embedding.

## License

Licensed under either the [Apache License, Version 2.0](LICENSE-APACHE) or
the [MIT license](LICENSE-MIT), at your option.

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in this project is licensed under the same terms.
