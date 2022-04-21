pub mod add_event_listener_5;
pub mod document_1;
pub mod on_submit_1;
pub mod window_0;

use std::cell::RefCell;
use std::rc::Rc;

use wasm_bindgen::closure::Closure;
use wasm_bindgen::JsCast;

use web_sys::{Event, EventTarget, Window};

use liblumen_alloc::erts::fragment::HeapFragment;
use liblumen_alloc::erts::term::prelude::*;

use crate::event_listener;
use crate::r#async;
use crate::runtime::process::spawn::options::Options;

pub fn add_event_listener(
    window: &Window,
    event: &'static str,
    module: Atom,
    function: Atom,
    options: Options,
) {
    let f = Rc::new(RefCell::new(None));
    let g = f.clone();

    let event_listener = move |event: &Event| {
        event.prevent_default();

        let promise_module = event_listener::module();
        let promise_function = event_listener::apply_4::function();
        let (event_listener_boxed_resource, event_listener_non_null_heap_fragment) =
            HeapFragment::new_resource(f.clone()).unwrap();
        let (event_boxed_resource, event_non_null_heap_fragment) =
            HeapFragment::new_resource(event.clone()).unwrap();
        let promise_argument_vec = vec![
            event_listener_boxed_resource.into(),
            event_boxed_resource.into(),
            module.encode().unwrap(),
            function.encode().unwrap(),
        ];

        let promise = r#async::apply_3::promise(
            promise_module,
            promise_function,
            promise_argument_vec,
            options,
        )
        .unwrap();

        // drop heap fragments now that term are cloned to spawned process in
        // reverse order
        std::mem::drop(event_non_null_heap_fragment);
        std::mem::drop(event_listener_non_null_heap_fragment);

        promise
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

pub fn module() -> Atom {
    Atom::from_str("Elixir.Lumen.Web.Window")
}
