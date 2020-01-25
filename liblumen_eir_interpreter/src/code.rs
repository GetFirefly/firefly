pub mod apply_3;
pub mod interpreter_closure;
pub mod interpreter_mfa;
pub mod return_clean;
pub mod return_ok;
pub mod return_throw;

use liblumen_alloc::erts::term::prelude::*;

fn module() -> Atom {
    Atom::try_from_str("lumen_eir_interpreter_intrinsics").unwrap()
}
