use std::path::PathBuf;
use std::process::Command;

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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Only needed when the embed feature is enabled.
    if std::env::var("CARGO_FEATURE_LEAN_EMBED").ok().is_none() {
        return Ok(());
    }

    // macOS: embed an rpath so `libleanshared.dylib` is found at runtime.
    if std::env::var("CARGO_CFG_TARGET_OS").ok().as_deref() == Some("macos") {
        let lean = resolve_tool("LEAN", "lean");
        let prefix = output(Command::new(&lean).arg("--print-prefix"))
            .or_else(|_| output(Command::new(&lean).arg("-print-prefix")))
            .map(PathBuf::from)
            .map_err(|e| format!("failed to get Lean prefix for rpath: {e}"))?;
        let lib_dir = prefix.join("lib").join("lean");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
    }

    Ok(())
}

