default: help

## proofpatch local workflows (CLI + MCP)

help:
    @echo "proofpatch (repo-local)"
    @echo ""
    @echo "Common:"
    @echo "  just test           # run Rust tests"
    @echo "  just build          # build proofpatch CLI + MCP"
    @echo "  just cli-help       # show proofpatch CLI help"
    @echo "  just mcp-help       # show proofpatch-mcp help"
    @echo "  just mcp-stdio      # run MCP server in stdio mode"
    @echo ""
    @echo "Tip: install a fast local binary:"
    @echo "  cargo build -p proofpatch-core --bin proofpatch --release"

test:
    cargo test -q

build:
    cargo build -q -p proofpatch-core --bin proofpatch
    cargo build -q -p proofpatch-mcp --bin proofpatch-mcp

build-release:
    cargo build -q -p proofpatch-core --bin proofpatch --release
    cargo build -q -p proofpatch-mcp --bin proofpatch-mcp --release

cli-help:
    cargo run -q -p proofpatch-core --bin proofpatch -- --help

mcp-help:
    cargo run -q -p proofpatch-mcp --bin proofpatch-mcp -- --help

mcp-stdio:
    cargo run -q -p proofpatch-mcp --bin proofpatch-mcp -- mcp-stdio

