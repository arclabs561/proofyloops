## `proofpatch-lean-embed`

Feature-gated **Lean 4 runtime embedding** for `proofpatch` (Rust), via the C ABI.

### Why this exists

This crate exists to make it easy to incrementally move “hard to fake correctly” logic into Lean:

- elaboration-aware classification of diagnostics
- robust syntax/tree inspection (beyond regex)
- (eventually) **proof tree searching** and proof-term structure queries

while keeping Rust responsible for:

- filesystem/process orchestration (`lake env lean`, timeouts, caching)
- stable JSON schemas + CLI/MCP surfaces

### Build/run (smoke test)

From `proofpatch/`:

```bash
cargo run -p proofpatch-core --features lean-embed --bin proofpatch -- lean-embed-smoke
```

Expected output:

```json
{"ok":true,"kind":"lean_embed_smoke","result":42,"expected":42}
```

### Extending toward proof-tree searching

Next useful steps (in order):

1. Export a Lean function that classifies a Lean diagnostic into a small enum-like `UInt8`
2. Export a Lean function that returns a small JSON string (Lean `String`) describing a “proof tree slice”
3. Add Rust wrappers that convert the result into `serde_json::Value` and plug into `triage-file`

