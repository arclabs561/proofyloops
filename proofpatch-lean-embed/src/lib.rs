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
/// Safe to call multiple times; initialization is global and one-time.
#[cfg(not(feature = "enabled"))]
pub fn init() -> Result<(), String> {
    Err("proofpatch-lean-embed is disabled (enable feature `proofpatch-lean-embed/enabled`)".to_string())
}

/// Initialize the embedded Lean runtime and the `LeanEmbedDemo.FFI` module.
///
/// Safe to call multiple times; initialization is global and one-time.
#[cfg(feature = "enabled")]
pub fn init() -> Result<(), String> {
    INIT.get_or_try_init(|| unsafe {
        let argv0 = CString::new("proofpatch-lean-embed").map_err(|e| e.to_string())?;
        let mut argv: Vec<*mut c_char> = vec![argv0.into_raw(), ptr::null_mut()];
        let _argv = lean_setup_args(1, argv.as_mut_ptr());

        lean_initialize_runtime_module();
        lean_initialize();

        let res = initialize_LeanEmbedDemo_FFI(1);
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
    Err("proofpatch-lean-embed is disabled (enable feature `proofpatch-lean-embed/enabled`)".to_string())
}

/// Smoke-test function: calls Lean-exported `pp_add_u64`.
#[cfg(feature = "enabled")]
pub fn add_u64(a: u64, b: u64) -> Result<u64, String> {
    init()?;
    Ok(unsafe { pp_add_u64(a, b) })
}

