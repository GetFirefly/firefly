//! Attempts to resolve the Promise attached to the `executor` by calling
//! `erlang:apply(module, function, arguments)` and using the returned value from `erlang:apply/3`
//! as the resolved value.
//!
//! ```elixir
//! Lumen.Web.Executor.apply(executor, module, function, arguments)
//! ```

mod label_1;

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use liblumen_otp::erlang;

#[native_implemented::function(Elixir.Lumen.Web.Executor:apply/4)]
pub fn result(
    process: &Process,
    executor: Term,
    module: Term,
    function: Term,
    arguments: Term,
) -> Term {
    process.queue_frame_with_arguments(
        erlang::apply_3::frame().with_arguments(false, &[module, function, arguments]),
    );
    process.queue_frame_with_arguments(label_1::frame().with_arguments(true, &[executor]));

    Term::NONE
}
