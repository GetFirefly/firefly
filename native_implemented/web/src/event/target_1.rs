//! ```elixir
//! case Lumen.Web.Event.target(event) do
//!
//! end
//! ```

use liblumen_alloc::atom;
use liblumen_alloc::erts::exception;
use liblumen_rt::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::event;

#[native_implemented::function(Elixir.Lumen.Web.Event:target/1)]
fn result(process: &Process, event_term: Term) -> exception::Result<Term> {
    let event = event::from_term(event_term)?;

    let final_term = match event.target() {
        Some(event_target) => {
            let event_target_resource_reference = process.resource(event_target);

            process.tuple_from_slice(&[atom!("ok"), event_target_resource_reference])
        }
        None => atom!("error"),
    };

    Ok(final_term)
}
