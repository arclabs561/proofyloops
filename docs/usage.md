# proofyloops usage

`proofyloops` is designed to run against any Lean 4 repo by pointing at a repo root and a target file/decl/region.

## CLI conventions

- **Repo root**: pass an absolute path with `--repo /abs/path/to/lean-repo`.
- **Outputs**: commands print JSON to stdout; many commands also support writing artifacts under `<repo_root>/.generated/…`.
- **Early exits**: many commands return quickly when work is unnecessary (e.g. “no `sorry` found”).

These local commands do not require an MCP server or a model account.

### Verification results and side effects

`triage-file` and `verify-summary` do not load `.env` files, Cursor MCP
credentials, or sibling-project credentials. They use the environment explicitly
passed to the process. This does not apply to optional model/research commands or
other legacy commands that still resolve credentials.

Verification runs Lean and may build missing dependencies. For a prepared checkout,
set `PROOFYLOOPS_AUTO_BUILD=0` to disable automatic `lake build`. This is not a
network sandbox: Lake and the target project's build configuration still control
their own dependency behavior. Select `PROOFYLOOPS_VERIFY_BACKEND=lake` for a direct
`lake env lean` invocation, and use `--timeout-s` to bound each verification step.

Exit status zero means the command produced a result, not that the proof is complete.
Read `.verify.summary.ok`, `.verify.summary.timeout`, and the diagnostic counts.
Lean accepts admitted declarations with a warning, so `ok: true` can coexist with
`sorry_warnings > 0`. `triage-file` also reports located holes in `.sorries`.

For a shell gate that rejects compilation errors, admission warnings, and located holes:

```bash
proofyloops triage-file --repo /abs/path/to/lean-repo --file Some/File.lean \
  | jq -e '.verify.summary.ok and (.verify.summary.counts.sorry_warnings == 0) and (.sorries.count == 0)'
```

This is a compilation/completeness check, not an axiom audit. Use Lean's
`#print axioms` separately when auditing a theorem's trust assumptions.

Triage reports `retry_verification` on timeout and
`inspect_verification_failure` when the verifier fails without a parsed Lean
error. Neither case means the file is complete. Timeout cancellation requests
termination of the directly spawned verifier process; it is not a sandbox or
a guarantee that every descendant process has stopped.

## Common commands

### Verify + scan for `sorry`

```bash
proofyloops triage-file --repo /abs/path/to/lean-repo --file Some/File.lean
```

### Verify a file (summary)

```bash
proofyloops verify-summary --repo /abs/path/to/lean-repo --file Some/File.lean
```

### Extract a bounded context pack

```bash
proofyloops context-pack --repo /abs/path/to/lean-repo --file Some/File.lean --decl some_theorem
```

### Create a scratch lemma file

```bash
proofyloops scratch-lemma --repo /abs/path/to/lean-repo --file Some/File.lean --name my_lemma
```

### Patch a lemma (in-memory) and verify

```bash
proofyloops patch --repo /abs/path/to/lean-repo --file Some/File.lean --lemma some_theorem --replacement-file /tmp/replacement.lean
```

## Repo-owned research suggestions

Put optional declaration selectors beside their research queries in the target
repository's `proofyloops.toml`. Prompts and `report` use the same selector;
the selector only describes possible research and never starts a search, model
request, or credential lookup. Some legacy report commands have their own
environment-loading behavior; selector choice does not add to it.

```toml
[research.presets.example]
query = "the mathematical result and useful proof vocabulary"
when_decl_contains = ["target_theorem", "target_namespace"]
```

Selectors are declaration-name substrings. Empty selectors are ignored. When
more than one selector matches, the longest selector wins; equal lengths use
the lexicographically smallest preset name. Without a match, Proofyloops emits
the generic `Lean proof <declaration>` suggestion.

## Focus controls

When a file has multiple `sorry`s, you can pin the search to one declaration:

```bash
proofyloops tree-search-nearest --repo /abs/path/to/lean-repo --file Some/File.lean --focus-decl MyNamespace.my_lemma
```

More strict variants:

- `--focus-decl-hard`: avoid drifting to other decls.
- `--focus-decl-strict`: fail fast if the decl does not match any `sorry` location.

## Output stability

Many commands include a stable `result_kind` string (e.g. `early_no_sorries`, `search`, `solved`) so downstream tooling can branch without brittle text matching.

For `tree-search-nearest`, inspect the selected result under `.picked`, not
the small `ok: true` receipt printed after writing an output file. A search
can finish successfully without finding a complete proof:

```bash
jq -e '.picked.verify.summary.ok and
  (.picked.verify.summary.counts.sorry_warnings == 0) and
  (.picked.sorries == 0)' search.json
```

## Command grouping aliases

These are equivalent:

- `proofyloops smt probe` == `proofyloops smt-probe`
- `proofyloops smt repro` == `proofyloops smt-repro`
