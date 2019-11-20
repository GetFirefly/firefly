// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::Term;

use lumen_runtime_macros::native_implemented_function;

use crate::otp::erlang::is_record;

#[native_implemented_function(is_record/3)]
pub fn native(
    process: &Process,
    term: Term,
    record_tag: Term,
    size: Term,
) -> exception::Result<Term> {
    is_record(process, term, record_tag, Some(size))
}
