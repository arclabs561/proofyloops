# proofyloops

`proofyloops` is a CLI + MCP server for **debuggable Lean 4 workflows**: verify, locate `sorry`s, extract bounded context packs, and (optionally) call an OpenAI-compatible LLM.

For tactic-level proof search or dataset extraction, use Pantograph or
LeanDojo; for an MCP view of Lean's language server (goals, diagnostics,
hovers), use lean-lsp-mcp. proofyloops works one level up, on files and
declarations: build them, find their `sorry`s, and patch them.

### Design

- **Target-agnostic**: point at a Lean repo with `--repo`, then target a file/decl/region inside it.
- **Evidence-first**: commands return structured JSON and can emit on-disk artifacts under `.generated/`.

### Quickstart (CLI)

From this repo:

```bash
cargo run -p proofyloops --bin proofyloops -- --help
```

Typical command:

```bash
cargo run -p proofyloops --bin proofyloops -- triage-file \
  --repo /abs/path/to/lean-repo \
  --file Some/File.lean
```

No MCP client or model account is needed for local triage, verification, or context
packs. To install the CLI from this checkout:

```bash
cargo install --path proofyloops-cli
```

See [CLI usage](docs/usage.md) for checking proof completeness, controlling builds,
and interpreting JSON results.

### Migrating from Proofpatch

The CLI is now `proofyloops`; the server is `proofyloops-mcp`. Reinstall from
this checkout and update executable paths in your MCP client, then restart it
to discover the renamed `proofyloops` and `proofyloops_*` tools.

Use `proofyloops.toml` and `PROOFYLOOPS_*` settings. The old `proofpatch.toml`
filename and `PROOFPATCH_*` runtime environment variables remain fallbacks;
the new names take precedence. Rust package names and the Lean helper import
are now `proofyloops-core` and `ProofyloopsTools`, respectively. Existing
published packages are not renamed automatically.

### MCP server

```bash
cargo run -p proofyloops-mcp --bin proofyloops-mcp
```

### SMT (optional oracle)

`proofyloops` can use `smtkit` as a **heuristic signal** (never as a proof) for LIA entailment checks.

### Quickstart: `tree-search-nearest` with SMT evidence

If you have an SMT solver on PATH (e.g. `z3`), a good “debuggable” run is:

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

- `--goal-dump` + `--smt-precheck` makes SMT checks much more informative (it extracts hypotheses/targets).
- `--smt-dump` writes bounded `.smt2` scripts to disk for inspection.
- `--smt-repro-dir` writes a self-contained repro bundle you can re-run with `proofyloops smt-repro`.

See `docs/smt.md` for:
- how dumping/repro bundles work
- UNSAT core / proof capture knobs
- `smt-repro` usage

### More docs

- `docs/usage.md`: common CLI patterns, focus flags, output stability.
- `lean-tools/README.md`: `ProofyloopsTools` (Lean side helper tactics).
- `proofyloops-lean-embed/README.md`: optional Lean runtime embedding.

## License

Licensed under either the [Apache License, Version 2.0](LICENSE-APACHE) or
the [MIT license](LICENSE-MIT), at your option.

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in this project is licensed under the same terms.
