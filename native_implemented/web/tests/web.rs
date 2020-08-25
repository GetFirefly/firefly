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

use liblumen_alloc::erts::apply::InitializeLumenDispatchTable;
use liblumen_alloc::erts::term::prelude::*;

use liblumen_web::r#async;
use liblumen_web::runtime;
use liblumen_web::runtime::process::spawn::options::Options;

wasm_bindgen_test_configure!(run_in_browser);

static START: Once = Once::new();

fn initialize_dispatch_table() {
    let function_symbols = vec![
        // Library
        liblumen_web::document::new_0::function_symbol(),
        liblumen_web::executor::apply_4::function_symbol(),
        liblumen_web::web_socket::new_1::function_symbol(),

        // Test
        document::body_1::with_body::function_symbol(),
        document::body_1::without_body::function_symbol(),
        element::class_name_1::test_0::function_symbol(),
        element::remove_1::removes_element::function_symbol(),
        math::random_integer_1::returns_integer_between_0_inclusive_and_max_exclusive::function_symbol(),
        node::insert_before_3::with_nil_reference_child_appends_new_child::function_symbol(),
        node::insert_before_3::with_reference_child_inserts_before_reference_child::function_symbol(),
        node::replace_child_3::with_new_child_is_parent_returns_error_hierarchy_request::function_symbol(),
        node::replace_child_3::with_new_child_returns_ok_replaced_child::function_symbol()
    ];

    unsafe {
        InitializeLumenDispatchTable(function_symbols.as_ptr(), function_symbols.len());
    }
}

fn start_once() {
    START.call_once(|| {
        initialize_dispatch_table();
        liblumen_web::start();
    })
}
