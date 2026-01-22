use std::path::PathBuf;
use std::process::Command;

fn run(cmd: &mut Command) -> Result<(), String> {
    let out = cmd
        .output()
        .map_err(|e| format!("failed to spawn {:?}: {e}", cmd))?;
    if !out.status.success() {
        return Err(format!(
            "command failed: {:?}\nstatus: {}\nstdout:\n{}\nstderr:\n{}",
            cmd,
            out.status,
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr),
        ));
    }
    Ok(())
}

fn output(cmd: &mut Command) -> Result<String, String> {
    let out = cmd
        .output()
        .map_err(|e| format!("failed to spawn {:?}: {e}", cmd))?;
    if !out.status.success() {
        return Err(format!(
            "command failed: {:?}\nstatus: {}\nstdout:\n{}\nstderr:\n{}",
            cmd,
            out.status,
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr),
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn home_dir() -> Option<PathBuf> {
    std::env::var("HOME")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
}

fn resolve_tool(env_key: &str, fallback_basename: &str) -> PathBuf {
    if let Ok(v) = std::env::var(env_key) {
        let v = v.trim().to_string();
        if !v.is_empty() {
            return PathBuf::from(v);
        }
    }
    if let Some(home) = home_dir() {
        let p = home.join(".elan").join("bin").join(fallback_basename);
        if p.exists() {
            return p;
        }
    }
    PathBuf::from(fallback_basename)
}

fn first_existing(paths: &[PathBuf]) -> Option<PathBuf> {
    for p in paths {
        if p.exists() {
            return Some(p.clone());
        }
    }
    None
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Keep default workspace builds fast: only do the heavy Lean build/link work when the
    // feature is explicitly enabled by a dependent crate.
    if std::env::var("CARGO_FEATURE_ENABLED").ok().is_none() {
        return Ok(());
    }

    println!("cargo:rerun-if-changed=lean/lean-toolchain");
    println!("cargo:rerun-if-changed=lean/lakefile.lean");
    println!("cargo:rerun-if-changed=lean/LeanEmbedDemo/FFI.lean");
    println!("cargo:rerun-if-changed=lean_shim.c");

    let lean_root = PathBuf::from("lean");
    let lake = resolve_tool("LAKE", "lake");
    let lean = resolve_tool("LEAN", "lean");

    // Build Lean IR, including generated C file.
    run(
        Command::new(&lake)
            .current_dir(&lean_root)
            .arg("build")
            .arg("LeanEmbedDemo.FFI"),
    )
    .map_err(|e| format!("lake build failed: {e}"))?;

    // We compile the generated C ourselves to get a stable `.a` for Rust to link.
    let c_file = first_existing(&[lean_root.join(".lake/build/ir/LeanEmbedDemo/FFI.c")])
        .ok_or_else(|| "could not find generated IR C file: .lake/build/ir/LeanEmbedDemo/FFI.c".to_string())?;

    let prefix = output(Command::new(&lean).arg("--print-prefix"))
        .or_else(|_| output(Command::new(&lean).arg("-print-prefix")))
        .map(PathBuf::from)
        .map_err(|e| format!("failed to get Lean prefix (need a working `lean`): {e}"))?;

    let include_dir = prefix.join("include");
    println!("cargo:rustc-env=LEAN_INCLUDE_DIR={}", include_dir.display());

    let lib_dir = prefix.join("lib").join("lean");
    println!("cargo:rustc-link-search=native={}", lib_dir.display());

    // macOS: make dependents runnable without DYLD_LIBRARY_PATH.
    if std::env::var("CARGO_CFG_TARGET_OS").ok().as_deref() == Some("macos") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
        println!("cargo:rustc-link-lib=dylib=c++");
    }

    // Compile FFI.c + shim into a static archive
    let mut cfg = cc::Build::new();
    cfg.file(&c_file);
    cfg.file("lean_shim.c");
    cfg.include(&include_dir);
    let gen_include = lean_root.join(".lake/build/include");
    if gen_include.exists() {
        cfg.include(&gen_include);
    }
    cfg.flag_if_supported("-std=c11");
    cfg.flag_if_supported("-mmacosx-version-min=11.0");
    cfg.compile("proofpatch_lean_embed");

    // Link against Lean runtime
    println!("cargo:rustc-link-lib=dylib=leanshared");
    println!("cargo:rustc-link-lib=static=leanrt");
    println!("cargo:rustc-link-lib=static=Init");

    Ok(())
}

