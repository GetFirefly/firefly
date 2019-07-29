#[cfg(target_arch = "wasm32")]
mod internal {
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen]
    extern "C" {
        #[wasm_bindgen(js_namespace = LumenStatsAlloc)]
        pub fn on_alloc(tag: String, size: usize, align: usize, ptr: *mut u8);

        #[wasm_bindgen(js_namespace = LumenStatsAlloc)]
        pub fn on_realloc(
            tag: String,
            old_size: usize,
            new_size: usize,
            align: usize,
            old_ptr: *mut u8,
            new_ptr: *mut u8,
        );

        #[wasm_bindgen(js_namespace = LumenStatsAlloc)]
        pub fn on_dealloc(tag: String, size: usize, align: usize, ptr: *mut u8);
    }
}

#[cfg(not(target_arch = "wasm32"))]
mod internal {
    #![allow(unused)]

    pub fn on_alloc(_tag: String, _size: usize, _align: usize, _ptr: *mut u8) {}

    pub fn on_realloc(
        _tag: String,
        _old_size: usize,
        _new_size: usize,
        _align: usize,
        _old_ptr: *mut u8,
        _new_ptr: *mut u8,
    ) {
    }

    pub fn on_dealloc(_tag: String, _size: usize, _align: usize, _ptr: *mut u8) {}
}

pub use internal::*;
