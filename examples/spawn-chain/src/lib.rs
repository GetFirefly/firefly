#![deny(warnings)]
#![feature(type_ascription)]

use std::sync::Arc;

use lumen_runtime::registry;
use lumen_runtime::scheduler::Scheduler;

use wasm_bindgen::prelude::*;

use crate::start::*;
use lumen_runtime::code::apply_fn;

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
    set_time_monotonic_source();
    set_apply_fn();

    let arc_scheduler = Scheduler::current();
    let init_arc_scheduler = Arc::clone(&arc_scheduler);
    let init_arc_process = init_arc_scheduler.spawn_init();
    let init_name = Term::str_to_atom("init", DoNotCare).unwrap();

    match Process::register(&init_arc_process, init_name) {
        Ok(_) => (),
        Err(_) => panic!("Could not register init process"),
    };
}

#[wasm_bindgen]
pub fn run(count: usize) {
    let init_arc_process = registry::name_to_process("init").unwrap();

    // elixir --erl "+P 1000000" -r chain.ex -e "Chain.run(1_000_000)".
    let module = Term::str_to_atom("Elixir.Chain", DoNotCare).unwrap();
    let function = Term::str_to_atom("run", DoNotCare).unwrap();
    let arguments =
        Term::slice_to_list(&[count.into_process(&init_arc_process)], &init_arc_process);
    let run_arc_process =
        Scheduler::spawn(&init_arc_process, module, function, arguments, apply_fn());

    loop {
        Scheduler::current().run_through(&run_arc_process);

        match *run_arc_process.status.read().unwrap() {
            Status::Exiting(ref exception) => match exception {
                Exception {
                    class: exception::Class::Exit,
                    reason,
                    ..
                } => {
                    if *reason != Term::str_to_atom("normal", DoNotCare).unwrap() {
                        #[cfg(debug_assertions)]
                        panic!("Process exited: {:?}", reason);
                        #[cfg(not(debug_assertions))]
                        panic!("Process exited");
                    } else {
                        break;
                    }
                }
                _ => {
                    #[cfg(debug_assertions)]
                    crate::code::print_stacktrace(&run_arc_process);
                    #[cfg(debug_assertions)]
                    panic!("Process exception: {:?}", exception);
                    #[cfg(not(debug_assertions))]
                    panic!("Process exception");
                }
            },
            Status::Waiting => {
                log_1(format!(
                    "WAITING Run queues len = {:?}",
                    Scheduler::current().run_queues_len()
                ));
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
