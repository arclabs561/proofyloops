fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::var("CARGO_FEATURE_LEAN_EMBED").is_err()
        || std::env::var("CARGO_CFG_TARGET_OS").ok().as_deref() != Some("macos")
    {
        return Ok(());
    }

    // The Lean embed crate discovers and publishes this path once. This final
    // executable is the layer that must retain it in LC_RPATH for dyld.
    let lib_dir = std::env::var("DEP_PROOFYLOOPS_LEAN_EMBED_LEAN_LIB_DIR").map_err(|_| {
        "Lean embed metadata was unavailable; rebuild with the `lean-embed` feature enabled"
    })?;
    println!("cargo:rustc-link-arg=-Wl,-rpath,{lib_dir}");
    Ok(())
}
