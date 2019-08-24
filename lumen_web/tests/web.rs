//! Test suite for the Web and headless browsers.
#![cfg(target_arch = "wasm32")]

#[path = "./web/document.rs"]
mod document;

extern crate wasm_bindgen_test;

use std::sync::Once;

use futures::future::Future;

use wasm_bindgen::JsValue;

use wasm_bindgen_futures::JsFuture;

use wasm_bindgen_test::*;

use liblumen_alloc::erts::process::code::stack::frame::Placement;

use lumen_runtime::process::spawn::options::Options;
use lumen_runtime::scheduler::Scheduler;

use lumen_web::wait;

wasm_bindgen_test_configure!(run_in_browser);

static START: Once = Once::new();

fn start_once() {
    START.call_once(|| {
        lumen_web::start();
    })
}
