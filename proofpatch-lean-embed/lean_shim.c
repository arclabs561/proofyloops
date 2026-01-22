#include <lean/lean.h>

// Expose a small subset of Lean's inline helpers as real symbols,
// so Rust can call them without pulling in the entire header surface.

LEAN_EXPORT uint8_t pp_lean_io_result_is_ok(b_lean_obj_arg r) {
  return lean_io_result_is_ok(r) ? 1 : 0;
}

LEAN_EXPORT void pp_lean_dec(b_lean_obj_arg o) {
  lean_dec(o);
}

LEAN_EXPORT void pp_lean_dec_ref(lean_object * o) {
  lean_dec_ref(o);
}

