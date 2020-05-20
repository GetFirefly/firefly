//! ```elixir
//! case Lumen.Web.Event.target(event) do
//!
//! end
//! ```

use liblumen_alloc::atom;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use native_implemented_function::native_implemented_function;

use crate::event;

#[native_implemented_function(target/1)]
fn result(process: &Process, event_term: Term) -> exception::Result<Term> {
    let event = event::from_term(event_term)?;

    match event.target() {
        Some(event_target) => {
            let event_target_resource_reference = process.resource(Box::new(event_target))?;

            process
                .tuple_from_slice(&[atom!("ok"), event_target_resource_reference])
                .map_err(|error| error.into())
        }
        None => Ok(atom!("error")),
    }
}
