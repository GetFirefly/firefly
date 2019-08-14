#![deny(warnings)]
// `Alloc`
#![feature(allocator_api)]
#![feature(type_ascription)]

mod apply_3;
mod elixir;
mod start;

use liblumen_alloc::erts::process::code::stack::frame::Placement;
use liblumen_alloc::erts::process::{heap, next_heap_size};

use lumen_runtime::scheduler::Scheduler;

use lumen_web::wait;

use wasm_bindgen::prelude::*;

use crate::elixir::chain::{console_1, dom_1};
use crate::start::*;

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global
// allocator.
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[wasm_bindgen(start)]
pub fn start() {
    set_panic_hook();
    set_apply_fn();
    lumen_web::start();
}

#[wasm_bindgen]
pub fn log_to_console(count: usize) -> js_sys::Promise {
    run(count, Output::Console)
}

#[wasm_bindgen]
pub fn log_to_dom(count: usize) -> js_sys::Promise {
    run(count, Output::Dom)
}

enum Output {
    Console,
    Dom,
}

fn run(count: usize, output: Output) -> js_sys::Promise {
    let arc_scheduler = Scheduler::current();
    // Don't register, so that tests can run concurrently
    let parent_arc_process = arc_scheduler.spawn_init(0).unwrap();

    let heap_size = next_heap_size(79 + count * 5);
    // if this fails the entire tab is out-of-memory
    let heap = heap(heap_size).unwrap();

    wait::with_return_0::spawn(
        &parent_arc_process,
        heap,
        heap_size,
    |child_process| {
            let count_term = child_process.integer(count)?;

            match output {
                Output::Console => {

                    // if this fails use a bigger sized heap
                    console_1::place_frame_with_arguments(child_process, Placement::Push, count_term)
                }
                Output::Dom => {
                    // if this fails use a bigger sized heap
                    dom_1::place_frame_with_arguments(child_process, Placement::Push, count_term)
                }
            }
        })
    // if this fails use a bigger sized heap
    .unwrap()
}
