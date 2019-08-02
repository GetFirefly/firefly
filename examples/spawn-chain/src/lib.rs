#![deny(warnings)]
// `AllocErr`
#![feature(allocator_api)]
#![feature(type_ascription)]

use std::sync::Arc;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::{heap, next_heap_size, Status};
use liblumen_alloc::erts::term::{atom_unchecked, Atom};

use lumen_runtime::code::apply_fn;
use lumen_runtime::registry;
use lumen_runtime::scheduler::Scheduler;

use wasm_bindgen::prelude::*;

use crate::start::*;

mod code;
mod elixir;
mod start;

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global
// allocator.
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[wasm_bindgen(start)]
pub fn start() {
    set_panic_hook();
    set_parking_lot_time_now_fn();
    set_apply_fn();

    let arc_scheduler = Scheduler::current();
    let init_arc_scheduler = Arc::clone(&arc_scheduler);
    let init_arc_process = init_arc_scheduler.spawn_init(0).unwrap();
    let init_atom = Atom::try_from_str("init").unwrap();

    if !registry::put_atom_to_process(init_atom, init_arc_process) {
        panic!("Could not register init process");
    };
}

#[wasm_bindgen]
pub fn run(count: usize) {
    let init_atom = Atom::try_from_str("init").unwrap();
    let init_arc_process = registry::atom_to_process(&init_atom).unwrap();

    // elixir --erl "+P 1000000" -r chain.ex -e "Chain.run(1_000_000)".
    let module = Atom::try_from_str("Elixir.Chain").unwrap();
    let function = Atom::try_from_str("run").unwrap();
    let arguments = init_arc_process
        .list_from_slice(&[init_arc_process.integer(count).unwrap()])
        // if not enough memory here, resize `spawn_init` heap
        .unwrap();

    let heap_size = next_heap_size(4 + count * 2);
    // if this fails the entire tab is out-of-memory
    let heap = heap(heap_size).unwrap();

    let run_arc_process = Scheduler::spawn(
        &init_arc_process,
        module,
        function,
        arguments,
        apply_fn(),
        heap,
        heap_size,
        )
        // if this fails, don't use `default_heap` and instead use a bigger sized heap
        .unwrap();

    loop {
        let ran = Scheduler::current().run_through(&run_arc_process);

        match *run_arc_process.status.read() {
            Status::Exiting(ref exception) => match exception {
                exception::runtime::Exception {
                    class: exception::runtime::Class::Exit,
                    reason,
                    ..
                } => {
                    if *reason != atom_unchecked("normal") {
                        #[cfg(debug_assertions)]
                        panic!("ProcessControlBlock exited: {:?}", reason);
                        #[cfg(not(debug_assertions))]
                        panic!("ProcessControlBlock exited");
                    } else {
                        break;
                    }
                }
                _ => {
                    #[cfg(debug_assertions)]
                    crate::code::print_stacktrace(&run_arc_process);
                    #[cfg(debug_assertions)]
                    panic!("ProcessControlBlock exception: {:?}", exception);
                    #[cfg(not(debug_assertions))]
                    panic!("ProcessControlBlock exception");
                }
            },
            Status::Waiting => {
                if ran {
                    log_1(format!(
                        "WAITING Run queues len = {:?}",
                        Scheduler::current().run_queues_len()
                    ));
                } else {
                    panic!(
                        "{:?} did not run.  Deadlock likely in {:#?}",
                        run_arc_process,
                        Scheduler::current()
                    );
                }
            }
            Status::Runnable => {
                log_1(format!(
                    "RUNNABLE Run queues len = {:?}",
                    Scheduler::current().run_queues_len()
                ));
            }
            Status::Running => {
                log_1(format!(
                    "RUNNING Run queues len = {:?}",
                    Scheduler::current().run_queues_len()
                ));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use time_test::time_test;

    #[test]
    fn run1() {
        time_test!();
        start();
        run(1)
    }

    #[test]
    fn run2() {
        time_test!();
        start();
        run(2)
    }

    #[test]
    fn run4() {
        time_test!();
        start();
        run(4)
    }

    #[test]
    fn run8() {
        time_test!();
        start();
        run(8)
    }

    #[test]
    fn run16() {
        time_test!();
        start();
        run(16)
    }

    #[test]
    fn run32() {
        time_test!();
        start();
        run(32)
    }

    #[test]
    fn run64() {
        time_test!();
        start();
        run(64)
    }

    #[test]
    fn run128() {
        time_test!();
        start();
        run(128)
    }

    #[test]
    fn run256() {
        time_test!();
        start();
        run(256)
    }

    #[test]
    fn run512() {
        time_test!();
        start();
        run(512)
    }

    #[test]
    fn run1024() {
        time_test!();
        start();
        run(1024)
    }

    #[test]
    fn run2048() {
        time_test!();
        start();
        run(2048)
    }

    #[test]
    fn run4096() {
        time_test!();
        start();
        run(4096)
    }

    #[test]
    fn run8192() {
        time_test!();
        start();
        run(8192)
    }

    #[test]
    fn run16384() {
        time_test!();
        start();
        run(16384)
    }

    #[test]
    fn run32768() {
        time_test!();
        start();
        run(32768)
    }

    #[test]
    fn run65536() {
        time_test!();
        start();
        run(65536)
    }

    #[test]
    fn run131072() {
        time_test!();
        start();
        run(131072)
    }
}
