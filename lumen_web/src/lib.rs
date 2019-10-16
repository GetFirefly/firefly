#![deny(warnings)]

pub mod document;
pub mod element;
pub mod event;
pub mod html_form_element;
pub mod html_input_element;
pub mod math;
pub mod node;
pub mod wait;
pub mod web_socket;
pub mod window;

use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

use web_sys::{DomException, Window};

use liblumen_alloc::erts::exception::system::Alloc;
use liblumen_alloc::erts::process::code::stack::frame::Placement;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::{atom_unchecked, Term};

use lumen_runtime::scheduler::Scheduler;
use lumen_runtime::time::monotonic::time_in_milliseconds;
use lumen_runtime::time::Milliseconds;

use crate::window::add_event_listener;

/// Starts the scheduler loop.  It yield and reschedule itself using
/// [requestAnimationFrame](https://developer.mozilla.org/en-US/docs/Web/API/window/requestAnimationFrame).
pub fn start() {
    add_event_listeners();
    request_animation_frames();
}

// Private

const MILLISECONDS_PER_SECOND: u64 = 1000;
const FRAMES_PER_SECOND: u64 = 60;
const MILLISECONDS_PER_FRAME: Milliseconds = MILLISECONDS_PER_SECOND / FRAMES_PER_SECOND;

fn add_event_listeners() {
    let window = web_sys::window().unwrap();
    add_submit_listener(&window);
}

fn add_submit_listener(window: &Window) {
    add_event_listener(
        window,
        "submit",
        Default::default(),
        |child_process, event_resource_reference| {
            window::on_submit_1::place_frame_with_arguments(
                child_process,
                Placement::Push,
                event_resource_reference,
            )
        },
    );
}

fn error() -> Term {
    atom_unchecked("error")
}

fn error_tuple(process: &Process, js_value: JsValue) -> Result<Term, Alloc> {
    let error = error();
    let dom_exception = js_value.dyn_ref::<DomException>().unwrap();

    match dom_exception.name().as_ref() {
        "SyntaxError" => {
            let tag = atom_unchecked("syntax");
            let message = process.binary_from_str(&dom_exception.message())?;
            let reason = process.tuple_from_slice(&[tag, message])?;

            process.tuple_from_slice(&[error, reason])
        }
        name => unimplemented!(
            "Converting {} DomException: {}",
            name,
            dom_exception.message()
        ),
    }
}

fn ok() -> Term {
    atom_unchecked("ok")
}

fn ok_tuple(process: &Process, value: Box<dyn Any>) -> Result<Term, Alloc> {
    let ok = ok();
    let resource_term = process.resource(value)?;

    process.tuple_from_slice(&[ok, resource_term])
}

fn option_to_ok_tuple_or_error<T: 'static>(
    process: &Process,
    option: Option<T>,
) -> Result<Term, Alloc> {
    match option {
        Some(value) => ok_tuple(process, Box::new(value)),
        None => Ok(error()),
    }
}

fn request_animation_frame(f: &Closure<dyn FnMut()>) {
    web_sys::window()
        .unwrap()
        .request_animation_frame(f.as_ref().unchecked_ref())
        .unwrap();
}

fn request_animation_frames() {
    // Based on https://github.com/rustwasm/wasm-bindgen/blob/603d5742eeca2a7a978f13614de9282229d1835e/examples/request-animation-frame/src/lib.rs
    let f = Rc::new(RefCell::new(None));
    let g = f.clone();

    *g.borrow_mut() = Some(Closure::wrap(Box::new(move || {
        run_for_milliseconds(MILLISECONDS_PER_FRAME);

        // Schedule ourselves for another requestAnimationFrame callback.
        request_animation_frame(f.borrow().as_ref().unwrap());
    }) as Box<dyn FnMut()>));

    request_animation_frame(g.borrow().as_ref().unwrap());
}

fn run_for_milliseconds(duration: Milliseconds) {
    let scheduler = Scheduler::current();
    let timeout = time_in_milliseconds() + duration;

    while (time_in_milliseconds() < timeout) && scheduler.run_once() {}
}
