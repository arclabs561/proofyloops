#![doc = include_str!("../README.md")]

#[cfg(feature = "enabled")]
use libc::{c_char, c_int};
#[cfg(feature = "enabled")]
use once_cell::sync::OnceCell;
#[cfg(feature = "enabled")]
use std::ffi::CString;
#[cfg(feature = "enabled")]
use std::ptr;

#[repr(C)]
pub struct lean_object {
    _private: [u8; 0],
}

#[cfg(feature = "enabled")]
extern "C" {
    fn lean_setup_args(argc: c_int, argv: *mut *mut c_char) -> *mut *mut c_char;
    fn lean_initialize_runtime_module();
    fn lean_initialize();
    fn lean_io_mark_end_initialization();

    #[link_name = "initialize_proofyloops__lean__embed_LeanEmbedDemo_FFI"]
    fn initialize_LeanEmbedDemo_FFI(builtin: u8) -> *mut lean_object;
    fn lean_io_result_show_error(res: *mut lean_object);

    // Shims (wrapping inline helpers)
    fn pp_lean_io_result_is_ok(res: *mut lean_object) -> u8;
    fn pp_lean_dec_ref(o: *mut lean_object);
    fn pp_lean_dec(o: *mut lean_object);

    // Exported Lean function
    fn pp_add_u64(a: u64, b: u64) -> u64;
}

#[cfg(feature = "enabled")]
static INIT: OnceCell<()> = OnceCell::new();

/// Initialize the embedded Lean runtime and the `LeanEmbedDemo.FFI` module.
///
/// Returns an error when runtime embedding is disabled.
#[cfg(not(feature = "enabled"))]
pub fn init() -> Result<(), String> {
    Err(
        "proofyloops-lean-embed is disabled (enable feature `proofyloops-lean-embed/enabled`)"
            .to_string(),
    )
}

/// Initialize the embedded Lean runtime and the `LeanEmbedDemo.FFI` module.
///
/// Successful initialization is cached and concurrent attempts are serialized
/// by this crate's `OnceCell`. Failed attempts may be retried; this does not
/// establish that native initialization can safely be repeated after failure.
/// Lean/libuv initialization is process-global: this experimental embedding
/// assumes exclusive ownership of it, not coordination with other embedders.
#[cfg(feature = "enabled")]
pub fn init() -> Result<(), String> {
    // SAFETY: argv has argc = 1 valid, writable, NUL-terminated string followed
    // by a null pointer. The table lives through initialization; into_raw keeps
    // the string allocated for process lifetime. Lean 4.35's lean_setup_args
    // forwards to libuv; libuv 1.51's Unix implementation copies the table but
    // retains the original string for process-title writes. Do not reclaim it.
    // This lifetime argument is reference-version-specific: the floating Lean
    // toolchain and overrides need ABI/ownership review when changed. OnceCell
    // serializes local attempts, not other users of the native global runtime;
    // the process-global and failure-retry assumptions above remain unverified.
    INIT.get_or_try_init(|| unsafe {
        let argv0 = CString::new("proofyloops-lean-embed").map_err(|e| e.to_string())?;
        let mut argv: Vec<*mut c_char> = vec![argv0.into_raw(), ptr::null_mut()];
        let _argv = lean_setup_args(1, argv.as_mut_ptr());

        lean_initialize_runtime_module();
        lean_initialize();

        let res = initialize_LeanEmbedDemo_FFI(1);
        // The generated initializer returns an owned Lean IO result. Inspect
        // and report it before releasing it once with the matching Lean helper.
        if pp_lean_io_result_is_ok(res) != 0 {
            pp_lean_dec_ref(res);
        } else {
            lean_io_result_show_error(res);
            pp_lean_dec(res);
            return Err("Lean module init failed".to_string());
        }

        lean_io_mark_end_initialization();
        Ok(())
    })?;
    Ok(())
}

/// Smoke-test function: calls Lean-exported `pp_add_u64`.
#[cfg(not(feature = "enabled"))]
pub fn add_u64(_a: u64, _b: u64) -> Result<u64, String> {
    Err(
        "proofyloops-lean-embed is disabled (enable feature `proofyloops-lean-embed/enabled`)"
            .to_string(),
    )
}

/// Smoke-test function: calls Lean-exported `pp_add_u64`.
#[cfg(feature = "enabled")]
pub fn add_u64(a: u64, b: u64) -> Result<u64, String> {
    init()?;
    // SAFETY: the Lean export's generated C signature is
    // uint64_t pp_add_u64(uint64_t, uint64_t), matching these by-value scalars.
    // No borrowed pointers cross this call; init succeeded first, subject to
    // its documented native initialization assumptions.
    Ok(unsafe { pp_add_u64(a, b) })
}
