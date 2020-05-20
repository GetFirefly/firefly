pub mod target_1;

use std::convert::TryInto;
use std::mem;

use anyhow::*;
use web_sys::Event;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::term::prelude::*;

// Private

fn from_term(term: Term) -> Result<&'static Event, exception::Exception> {
    let boxed: Boxed<Resource> = term
        .try_into()
        .with_context(|| format!("{} must be an event resource", term))?;
    let event_reference: Resource = boxed.into();

    match event_reference.downcast_ref() {
        Some(event) => {
            let static_event: &'static Event = unsafe { mem::transmute::<&Event, _>(event) };

            Ok(static_event)
        }
        None => Err(TypeError)
            .with_context(|| format!("{} is a resource, but not an event", term))
            .map_err(From::from),
    }
}

fn module() -> Atom {
    Atom::try_from_str("Elixir.Lumen.Web.Event").unwrap()
}

fn module_id() -> usize {
    module().id()
}
