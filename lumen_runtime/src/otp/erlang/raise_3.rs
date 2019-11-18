// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::term::prelude::Term;
use liblumen_alloc::{badarg, raise};

use lumen_runtime_macros::native_implemented_function;

use crate::stacktrace;

#[native_implemented_function(raise/3)]
pub fn native(class: Term, reason: Term, stacktrace: Term) -> exception::Result<Term> {
    let class_class: exception::Class = class.try_into()?;

    let runtime_exception = if stacktrace::is(stacktrace) {
        raise!(class_class, reason, stacktrace)
    } else {
        badarg!()
    };

    Err(runtime_exception.into())
}
