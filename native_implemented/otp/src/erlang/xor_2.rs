#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::term::prelude::*;

/// `xor/2` infix operator.
///
/// **NOTE: NOT SHORT-CIRCUITING!**
#[native_implemented::function(xor/2)]
pub fn result(left_boolean: Term, right_boolean: Term) -> exception::Result<Term> {
    boolean_infix_operator!(left_boolean, right_boolean, ^)
}
