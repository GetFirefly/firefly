pub mod add_event_listener_5;
pub mod document_1;
pub mod on_submit_1;
pub mod window_0;

use std::cell::RefCell;
use std::rc::Rc;

use wasm_bindgen::closure::Closure;
use wasm_bindgen::JsCast;

use web_sys::{Event, EventTarget, Window};

use liblumen_alloc::erts::process::frames;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use lumen_rt_full::process::spawn::options::Options;

use crate::wait;

pub fn add_event_listener<F>(
    window: &Window,
    event: &'static str,
    options: Options,
    place_frame_with_arguments: F,
) where
    F: Fn(&Process, Term) -> frames::Result + 'static,
{
    let f = Rc::new(RefCell::new(None));
    let g = f.clone();

    let event_listener = move |event: &Event| {
        event.prevent_default();

        wait::with_return_0::spawn(options, |child_process| {
            // put reference to this closure into process dictionary so that it can't be GC'd until
            // `child_process` exits and is `Drop`'d.
            let event_listener_resource_reference = child_process.resource(Box::new(f.clone()))?;
            child_process
                .put(
                    Atom::str_to_term("Elixir.Lumen.Web.Window.event_listener"),
                    event_listener_resource_reference,
                )
                .unwrap();

            let event_resource_reference = child_process.resource(Box::new(event.clone()))?;

            place_frame_with_arguments(child_process, event_resource_reference)
        })
        .unwrap()
    };

    let event_listener_box: Box<dyn FnMut(&Event) -> js_sys::Promise> = Box::new(event_listener);
    let event_listener_closure = Closure::wrap(event_listener_box);

    *g.borrow_mut() = Some(event_listener_closure);

    let window_event_target: &EventTarget = window.as_ref();

    window_event_target
        .add_event_listener_with_callback(
            event,
            g.borrow().as_ref().unwrap().as_ref().unchecked_ref(),
        )
        .unwrap();
}

// Private

fn module() -> Atom {
    Atom::try_from_str("Elixir.Lumen.Web.Window").unwrap()
}
