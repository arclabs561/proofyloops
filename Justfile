default: help

## proofyloops local workflows (CLI + MCP)

help:
    @echo "proofyloops (repo-local)"
    @echo ""
    @echo "Common:"
    @echo "  just test           # run Rust tests"
    @echo "  just build          # build proofyloops CLI + MCP"
    @echo "  just cli-help       # show proofyloops CLI help"
    @echo "  just mcp-help       # show proofyloops-mcp help"
    @echo "  just mcp-stdio      # run MCP server in stdio mode"
    @echo ""
    @echo "Tip: install a fast local binary:"
    @echo "  cargo build -p proofyloops --bin proofyloops --release"

test:
    cargo test -q

build:
    cargo build -q -p proofyloops --bin proofyloops
    cargo build -q -p proofyloops-mcp --bin proofyloops-mcp

build-release:
    cargo build -q -p proofyloops --bin proofyloops --release
    cargo build -q -p proofyloops-mcp --bin proofyloops-mcp --release

cli-help:
    cargo run -q -p proofyloops --bin proofyloops -- --help

mcp-help:
    cargo run -q -p proofyloops-mcp --bin proofyloops-mcp -- --help

mcp-stdio:
    cargo run -q -p proofyloops-mcp --bin proofyloops-mcp -- mcp-stdio

