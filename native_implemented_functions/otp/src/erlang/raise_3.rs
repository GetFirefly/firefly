// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;

use anyhow::*;

use liblumen_alloc::erts::exception::{self, *};
use liblumen_alloc::erts::term::prelude::*;

use native_implemented_function::native_implemented_function;

use crate::runtime::stacktrace;

#[native_implemented_function(raise/3)]
pub fn result(class: Term, reason: Term, stacktrace: Term) -> exception::Result<Term> {
    let class_class: exception::Class = class.try_into()?;

    if stacktrace::is(stacktrace) {
        Err(raise(
            class_class,
            reason,
            Some(stacktrace),
            anyhow!("explicit raise from Erlang").into(),
        )
        .into())
    } else {
        Err(TypeError)
            .context(format!("stacktrace ({}) is not a stacktrace", stacktrace))
            .map_err(From::from)
    }
}
