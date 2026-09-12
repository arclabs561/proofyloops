#![cfg(unix)]

use proofyloops_core::verify_lean_file;
use std::ffi::OsString;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

struct EnvRestore(Vec<(&'static str, Option<OsString>)>);

impl EnvRestore {
    fn capture(names: &[&'static str]) -> Self {
        Self(
            names
                .iter()
                .map(|name| (*name, std::env::var_os(name)))
                .collect(),
        )
    }
}

impl Drop for EnvRestore {
    fn drop(&mut self) {
        for (name, value) in self.0.drain(..) {
            if let Some(value) = value {
                std::env::set_var(name, value);
            } else {
                std::env::remove_var(name);
            }
        }
    }
}

struct KillRecordedPid(PathBuf);

impl Drop for KillRecordedPid {
    fn drop(&mut self) {
        if let Ok(pid) = read_pid(&self.0) {
            let _ = std::process::Command::new("/bin/kill")
                .args(["-TERM", &pid.to_string()])
                .status();
        }
    }
}

fn read_pid(path: &Path) -> Result<u32, String> {
    fs::read_to_string(path)
        .map_err(|e| format!("read {}: {e}", path.display()))?
        .trim()
        .parse::<u32>()
        .map_err(|e| format!("parse pid from {}: {e}", path.display()))
}

fn pid_state(pid: u32) -> Option<String> {
    let output = std::process::Command::new("/bin/ps")
        .args(["-o", "state=", "-p", &pid.to_string()])
        .output()
        .ok()?;
    let state = String::from_utf8_lossy(&output.stdout).trim().to_string();
    (!state.is_empty()).then_some(state)
}

fn pid_is_running(pid: u32) -> bool {
    !matches!(pid_state(pid).as_deref(), None | Some("Z"))
}

#[test]
fn timed_lake_verification_terminates_the_direct_child() {
    let env_names = [
        "LAKE",
        "PROOFYLOOPS_AUTO_BUILD",
        "PROOFYLOOPS_VERIFY_BACKEND",
        "PROOFYLOOPS_TEST_PID_FILE",
    ];
    let _env_restore = EnvRestore::capture(&env_names);

    let td = tempfile::tempdir().expect("temporary test directory");
    let repo_root = td.path().join("synthetic-lean-repo");
    fs::create_dir_all(repo_root.join(".lake/build/lib/lean")).expect("build output directory");
    fs::write(repo_root.join("lakefile.lean"), "package synthetic\n").expect("lakefile");
    fs::write(repo_root.join("lean-toolchain"), "v4.0.0\n").expect("lean toolchain");
    fs::write(
        repo_root.join("Example.lean"),
        "theorem example : True := by trivial\n",
    )
    .expect("Lean input");

    let pid_file = td.path().join("fake-lake.pid");
    let _cleanup = KillRecordedPid(pid_file.clone());
    let fake_lake = td.path().join("fake-lake");
    fs::write(
        &fake_lake,
        "#!/bin/sh\nprintf '%s\\n' \"$$\" > \"$PROOFYLOOPS_TEST_PID_FILE\"\nexec /bin/sleep 5\n",
    )
    .expect("fake lake script");
    let mut permissions = fs::metadata(&fake_lake)
        .expect("fake lake metadata")
        .permissions();
    permissions.set_mode(0o700);
    fs::set_permissions(&fake_lake, permissions).expect("make fake lake executable");

    std::env::set_var("LAKE", &fake_lake);
    std::env::set_var("PROOFYLOOPS_AUTO_BUILD", "0");
    std::env::set_var("PROOFYLOOPS_VERIFY_BACKEND", "lake");
    std::env::set_var("PROOFYLOOPS_TEST_PID_FILE", &pid_file);
    assert_eq!(proofyloops_core::resolve_lake(), fake_lake);

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("test runtime");
    let result = runtime
        .block_on(verify_lean_file(
            &repo_root,
            "Example.lean",
            Duration::from_millis(500),
        ))
        .expect("verification result");
    assert!(
        result.timeout,
        "the intentionally slow fake lake must time out"
    );
    let pid = read_pid(&pid_file).expect("fake lake wrote its pid before timing out");
    let deadline = Instant::now() + Duration::from_millis(500);
    while pid_is_running(pid) && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(
        !pid_is_running(pid),
        "timing out verification must stop its direct lake child (pid {pid}, state {:?})",
        pid_state(pid)
    );
}
