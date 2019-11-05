//! ```elixir
//! case Lumen.Web.Event.target(event) do
//!
//! end
//! ```

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::Term;

use lumen_runtime_macros::native_implemented_function;

use crate::{error, event, ok};

#[native_implemented_function(target/1)]
fn native(process: &Process, event_term: Term) -> exception::Result {
    let event = event::from_term(event_term)?;

    match event.target() {
        Some(event_target) => {
            let event_target_resource_reference = process.resource(Box::new(event_target))?;

            process
                .tuple_from_slice(&[ok(), event_target_resource_reference])
                .map_err(|error| error.into())
        }
        None => Ok(error()),
    }
}
