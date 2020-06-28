//! ```elixir
//! case Lumen.Web.Window.add_event_listener(window, :submit, MyModule, :my_function) do
//!   :ok -> ...
//!   :error -> ...
//! end
//! ```

use std::convert::TryInto;

use anyhow::*;
use web_sys::Window;

use liblumen_alloc::atom;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::term::prelude::*;

use crate::runtime::process::spawn::options::Options;

use liblumen_otp::erlang;

use crate::window::add_event_listener;

#[native_implemented::function(add_event_listener/4)]
fn result(window: Term, event: Term, module: Term, function: Term) -> exception::Result<Term> {
    let boxed: Boxed<Resource> = window
        .try_into()
        .with_context(|| format!("window ({}) must be a window resource", window))?;
    let window_reference: Resource = boxed.into();
    let window_window: &Window = window_reference
        .downcast_ref()
        .with_context(|| format!("{} is a resource, but not a window", window))?;

    let event_atom: Atom = event
        .try_into()
        .with_context(|| format!("event ({}) must be an atom", event))?;
    let _: Atom = module
        .try_into()
        .with_context(|| format!("module ({}) must be an atom", module))?;
    let _: Atom = function
        .try_into()
        .with_context(|| format!("function ({}) must be an atom", function))?;

    // TODO support passing in options to allow bigger heaps
    let options: Options = Default::default();

    add_event_listener(
        window_window,
        event_atom.name(),
        options,
        move |child_process, event_resource_reference| {
            let arguments = child_process.list_from_slice(&[event_resource_reference])?;

            let frame_with_argument =
                erlang::apply_3::frame_with_arguments(module, function, arguments);

            Ok(vec![frame_with_argument])
        },
    );

    Ok(atom!("ok"))
}
