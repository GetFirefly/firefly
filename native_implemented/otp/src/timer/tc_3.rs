//! ```elixir
//! def tc(module, function, arguments) do
//!   before = :erlang.monotonic_time()
//!   value = apply(module, function, arguments)
//!   after = :erlang.monotonic_time()
//!   duration = after - before
//!   time = :erlang.convert_time_unit(duration, :native, :microsecond)
//!   {time, value}
//! end
//! ```

mod label_1;
mod label_2;
mod label_3;
mod label_4;
mod label_5;

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::erlang::monotonic_time_0;

// Private

#[native_implemented::function(timer:tc/3)]
fn result(process: &Process, module: Term, function: Term, arguments: Term) -> Term {
    process.queue_frame_with_arguments(monotonic_time_0::frame().with_arguments(false, &[]));
    process.queue_frame_with_arguments(
        label_1::frame().with_arguments(true, &[module, function, arguments]),
    );

    Term::NONE
}
