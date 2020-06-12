#![deny(warnings)]

//! Test suite for the Web and headless browsers.
#![cfg(target_arch = "wasm32")]

#[path = "web/document.rs"]
mod document;
#[path = "web/element.rs"]
mod element;
#[path = "web/math.rs"]
mod math;
#[path = "web/node.rs"]
mod node;
#[path = "web/web_socket.rs"]
mod web_socket;

extern crate wasm_bindgen_test;

use std::sync::Once;

use futures::future::Future;

use wasm_bindgen::JsValue;

use wasm_bindgen_futures::JsFuture;

use wasm_bindgen_test::*;

use liblumen_alloc::erts::process::frames::stack::frame::Placement;

use lumen_rt_full::process::spawn::options::Options;

use liblumen_web::wait;

wasm_bindgen_test_configure!(run_in_browser);

static START: Once = Once::new();

fn start_once() {
    START.call_once(|| {
        liblumen_web::start();
    })
}
