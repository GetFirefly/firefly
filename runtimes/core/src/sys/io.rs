#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console, js_name = log)]
    pub fn console_log(s: &str);
}

#[cfg(not(target_arch = "wasm32"))]
pub fn puts(s: &str) {
    println!("{}", s);
}

#[cfg(target_arch = "wasm32")]
pub fn puts(s: &str) {
    console_log(s);
}
