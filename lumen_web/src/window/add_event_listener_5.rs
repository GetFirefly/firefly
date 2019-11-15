//! ```elixir
//! case Lumen.Web.Window.add_event_listener(window, :submit, MyModule, :my_function) do
//!   :ok -> ...
//!   :error -> ...
//! end
//! ```

use std::convert::TryInto;

use web_sys::Window;

use liblumen_alloc::{badarg, atom};
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::code::stack::frame::Placement;
use liblumen_alloc::erts::term::prelude::*;

use lumen_runtime_macros::native_implemented_function;

use lumen_runtime::otp::erlang;
use lumen_runtime::process::spawn::options::Options;

use crate::window::add_event_listener;


#[native_implemented_function(add_event_listener/4)]
fn native(window: Term, event: Term, module: Term, function: Term) -> exception::Result<Term> {
    let boxed: Boxed<Resource> = window.try_into()?;
    let window_reference: Resource = boxed.into();
    let window_window: &Window = window_reference.downcast_ref().ok_or_else(|| badarg!())?;

    let event_atom: Atom = event.try_into()?;
    let _: Atom = module.try_into()?;
    let _: Atom = function.try_into()?;

    // TODO support passing in options to allow bigger heaps
    let options: Options = Default::default();

    add_event_listener(
        window_window,
        event_atom.name(),
        options,
        move |child_process, event_resource_reference| {
            let arguments = child_process.list_from_slice(&[event_resource_reference])?;

            erlang::apply_3::place_frame_with_arguments(
                child_process,
                Placement::Push,
                module,
                function,
                arguments,
            ).map_err(|e| e.into())
        },
    );

    Ok(atom!("ok"))
}
