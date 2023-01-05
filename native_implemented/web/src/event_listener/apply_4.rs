//! Run `module:function(event)` while keeping `event_listener` alive.
//!
//! ```elixir
//! Lumen.Web.EventListener.apply(event_listener, event, module, function)
//! ```
//!
//! The `event_listener` is passed along so that it stays alive while the `module:function(event)`
//! is run.

mod label_1;

use liblumen_rt::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use liblumen_otp::erlang;

#[native_implemented::function(Elixir.Lumen.Web.EventListener:apply/4)]
fn result(
    process: &Process,
    event_listener: Term,
    event: Term,
    module: Term,
    function: Term,
) -> Term {
    let arguments = process.list_from_slice(&[event]);

    process.queue_frame_with_arguments(
        erlang::apply_3::frame().with_arguments(false, &[module, function, arguments]),
    );
    process.queue_frame_with_arguments(label_1::frame().with_arguments(true, &[event_listener]));

    Term::NONE
}
