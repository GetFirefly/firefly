//! ```elixir
//! # label 1
//! # pushed to stack: (module, function arguments, before)
//! # returned from call: before
//! # full stack: (before, module, function arguments)
//! # returns: value
//! value = apply(module, function, arguments)
//! after = :erlang.monotonic_time()
//! duration = after - before
//! time = :erlang.convert_time_unit(duration, :native, :microsecond)
//! {time, value}
//! ```

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::erlang::apply_3;

use super::label_2;

// Private

#[native_implemented::label]
fn result(process: &Process, before: Term, module: Term, function: Term, arguments: Term) -> Term {
    assert!(before.is_integer());
    assert!(module.is_atom(), "module ({:?}) is not an atom", module);
    assert!(function.is_atom());
    assert!(arguments.is_list());

    process.queue_frame_with_arguments(
        apply_3::frame().with_arguments(false, &[module, function, arguments]),
    );
    process.queue_frame_with_arguments(label_2::frame().with_arguments(true, &[before]));

    Term::NONE
}
