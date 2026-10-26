# Changelog

## Source history (unversioned)

These entries identify source changes without assigning a package release.

- Preserve distinct SMT variable identities, including names such as `x.y` and `x_y`.
  ([b1b48c5](https://github.com/arclabs561/proofyloops/commit/b1b48c5cfa7e8ec6c0bbb21bd742abdc7fed125a))
- Clarify Lean FFI lifetime, initialization and process-global ownership assumptions in safety comments.
  ([1104298](https://github.com/arclabs561/proofyloops/commit/11042985b7e12c1a3f2128b3a8bedae8b79a754f))

## Unreleased

- Model credential lookup reads only `<repo_root>/.env` by default. Reading
  `~/.cursor/mcp.json` and parent or sibling `.env` files now requires
  `PROOFYLOOPS_DOTENV_SEARCH=1` (previously on unless set to 0).
- `~/.cursor/mcp.json` env blocks go through the same filter as `.env` files:
  endpoint, proxy and CA variables are ignored.
- Documentation-only changes.

