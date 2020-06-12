#[cfg(all(not(any(target_arch = "wasm32", feature = "runtime_minimal")), test))]
mod test;

use std::convert::TryInto;

use anyhow::*;

use liblumen_alloc::erts::exception::{self, *};
use liblumen_alloc::erts::term::prelude::*;

use crate::runtime::stacktrace;

#[native_implemented::function(raise/3)]
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
