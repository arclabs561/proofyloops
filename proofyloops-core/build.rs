fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Only needed when the embed feature is enabled.
    if std::env::var("CARGO_FEATURE_LEAN_EMBED").ok().is_none() {
        return Ok(());
    }

    // This crate can itself produce test executables. Receive the Lean library
    // location from the native-link owner rather than locating Lean again.
    if std::env::var("CARGO_CFG_TARGET_OS").ok().as_deref() == Some("macos") {
        let lib_dir = std::env::var("DEP_PROOFYLOOPS_LEAN_EMBED_LEAN_LIB_DIR").map_err(|_| {
            "Lean embed metadata was unavailable; rebuild with the `lean-embed` feature enabled"
        })?;
        println!("cargo:rustc-link-arg=-Wl,-rpath,{lib_dir}");
        // Forward native-link metadata to executable consumers such as the
        // MCP server when workspace feature unification enables embedding.
        println!("cargo:LEAN_LIB_DIR={lib_dir}");
    }

    Ok(())
}
