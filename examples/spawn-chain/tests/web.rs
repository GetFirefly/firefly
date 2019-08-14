//! Test suite for the Web and headless browsers.
#![cfg(target_arch = "wasm32")]

extern crate wasm_bindgen_test;

use std::sync::Once;

use wasm_bindgen_test::*;

use spawn_chain::start;

wasm_bindgen_test_configure!(run_in_browser);

mod log_to_console {
    use super::*;

    use spawn_chain::log_to_console;

    #[wasm_bindgen_test]
    fn with_1() {
        start_once();
        assert_eq!(log_to_console(1), 1);
    }

    #[wasm_bindgen_test]
    fn with_2() {
        start_once();
        assert_eq!(log_to_console(2), 2);
    }

    #[wasm_bindgen_test]
    fn with_4() {
        start_once();
        assert_eq!(log_to_console(4), 4);
    }

    #[wasm_bindgen_test]
    fn with_8() {
        start_once();
        assert_eq!(log_to_console(8), 8);
    }

    #[wasm_bindgen_test]
    fn with_16() {
        start_once();
        assert_eq!(log_to_console(16), 16);
    }

    #[wasm_bindgen_test]
    fn with_32() {
        start_once();
        assert_eq!(log_to_console(32), 32);
    }
}

mod log_to_dom {
    use super::*;

    use spawn_chain::log_to_dom;

    #[wasm_bindgen_test]
    fn with_1() {
        start_once();
        assert_eq!(log_to_dom(1), 1);
    }

    #[wasm_bindgen_test]
    fn with_2() {
        start_once();
        assert_eq!(log_to_dom(2), 2);
    }

    #[wasm_bindgen_test]
    fn with_4() {
        start_once();
        assert_eq!(log_to_dom(4), 4);
    }

    #[wasm_bindgen_test]
    fn with_8() {
        start_once();
        assert_eq!(log_to_dom(8), 8);
    }

    #[wasm_bindgen_test]
    fn with_16() {
        start_once();
        assert_eq!(log_to_dom(16), 16);
    }

    #[wasm_bindgen_test]
    fn with_32() {
        start_once();
        assert_eq!(log_to_dom(32), 32);
    }
}

static START: Once = Once::new();

fn start_once() {
    START.call_once(|| {
        start();
    })
}
