pub mod target_1;

use std::convert::TryInto;
use std::mem;

use web_sys::Event;

use liblumen_alloc::badarg;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::term::prelude::*;

// Private

fn from_term(term: Term) -> Result<&'static Event, exception::Exception> {
    let boxed: Boxed<Resource> = term.try_into()?;
    let event_reference: Resource = term.into();

    match event_reference.downcast_ref() {
        Some(event) => {
            let static_event: &'static Event = unsafe { mem::transmute::<&Event, _>(event) };

            Ok(static_event)
        }
        None => Err(badarg!().into()),
    }
}

fn module() -> Atom {
    Atom::try_from_str("Elixir.Lumen.Web.Event").unwrap()
}
