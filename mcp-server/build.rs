fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").ok().as_deref() != Some("macos") {
        return;
    }

    // `proofyloops-core` publishes this only when its `lean-embed` feature is
    // active. Normal MCP builds do not link the optional Lean runtime.
    if let Ok(lib_dir) = std::env::var("DEP_PROOFYLOOPS_CORE_LEAN_LIB_DIR") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{lib_dir}");
    }
}
