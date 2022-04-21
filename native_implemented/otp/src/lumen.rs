//! Lumen intrinsics

pub mod apply_apply_2_1;
pub mod apply_apply_3_1;
pub mod is_big_integer_1;
pub mod is_small_integer_1;
pub mod log_exit_1;

use liblumen_alloc::erts::term::prelude::*;

pub fn module() -> Atom {
    Atom::from_str("lumen")
}
