// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::term::prelude::Term;

use lumen_runtime_macros::native_implemented_function;

/// `orelse/2` infix operator.
///
/// Short-circuiting, but doesn't enforce `right` is boolean.  If you need to enforce `boolean` for
/// both operands, use `or_2`.
#[native_implemented_function(and/2)]
pub fn native(boolean: Term, term: Term) -> exception::Result<Term> {
    let boolean_bool: bool = boolean.try_into()?;

    if boolean_bool {
        // always `true.into()`, but this is faster
        Ok(boolean)
    } else {
        Ok(term)
    }
}
