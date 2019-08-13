#![deny(warnings)]
// `Alloc`
#![feature(allocator_api)]
#![feature(type_ascription)]

mod apply_3;
mod elixir;
mod start;

use std::convert::TryInto;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::code::stack::frame::Placement;
use liblumen_alloc::erts::process::{heap, next_heap_size, Status};

use lumen_runtime::scheduler::Scheduler;
use lumen_runtime::system;

use lumen_web::wait;

use wasm_bindgen::prelude::*;

use crate::elixir::chain::{console_1, dom_1};
use crate::start::*;
use liblumen_alloc::erts::term::atom_unchecked;

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global
// allocator.
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[wasm_bindgen(start)]
pub fn start() {
    set_panic_hook();
    set_apply_fn();
}

#[wasm_bindgen]
pub fn log_to_console(count: usize) -> usize {
    run(count, Output::Console)
}

#[wasm_bindgen]
pub fn log_to_dom(count: usize) -> usize {
    run(count, Output::Dom)
}

enum Output {
    Console,
    Dom,
}

fn run(count: usize, output: Output) -> usize {
    let arc_scheduler = Scheduler::current();
    // Don't register, so that tests can run concurrently
    let parent_arc_process = arc_scheduler.spawn_init(0).unwrap();

    // if not enough memory here, resize `spawn_init` heap
    let count_term = parent_arc_process.integer(count).unwrap();

    let heap_size = next_heap_size(79 + count * 5);
    // if this fails the entire tab is out-of-memory
    let heap = heap(heap_size).unwrap();

    let run_arc_process = wait::with_return_0::spawn(
        &parent_arc_process,
        heap,
        heap_size
    )
    // if this fails use a bigger sized heap
    .unwrap();

    match output {
        Output::Console => {
            // if this fails use a bigger sized heap
            console_1::place_frame_with_arguments(&run_arc_process, Placement::Push, count_term)
                .unwrap()
        }
        Output::Dom => {
            // if this fails use a bigger sized heap
            dom_1::place_frame_with_arguments(&run_arc_process, Placement::Push, count_term)
                .unwrap()
        }
    };

    let mut option_return_usize: Option<usize> = None;

    loop {
        let ran = Scheduler::current().run_through(&run_arc_process);

        let waiting = match *run_arc_process.status.read() {
            Status::Exiting(ref exception) => match exception {
                exception::runtime::Exception {
                    class: exception::runtime::Class::Exit,
                    reason,
                    ..
                } => {
                    if *reason != atom_unchecked("normal") {
                        panic!("ProcessControlBlock exited: {:?}", reason);
                    } else {
                        break;
                    }
                }
                _ => {
                    panic!(
                        "ProcessControlBlock exception: {:?}\n{:?}",
                        exception,
                        run_arc_process.stacktrace()
                    );
                }
            },
            Status::Waiting => true,
            Status::Runnable => false,
            Status::Running => {
                system::io::puts(&format!(
                    "RUNNING Run queues len = {:?}",
                    Scheduler::current().run_queues_len()
                ));

                false
            }
        };

        // separate so we don't hold read lock on status as it may need to be written
        if waiting {
            if ran {
                system::io::puts(&format!(
                    "WAITING Run queues len = {:?}",
                    Scheduler::current().run_queues_len()
                ));
            } else {
                use wait::with_return_0::Error::*;

                match wait::with_return_0::pop_return(&run_arc_process) {
                    Ok(popped_return) => {
                        option_return_usize = Some(popped_return.try_into().unwrap());
                        wait::with_return_0::stop(&run_arc_process);
                    },
                    Err(NoModuleFunctionArity) => panic!("{:?} doesn't have a current module function arity", run_arc_process),
                    Err(WrongModuleFunctionArity(current_module_function_arity)) => panic!(
                        "{:?} is not waiting with a return and instead did not run while waiting in {}.  Deadlock likely in {:#?}",
                        run_arc_process,
                        current_module_function_arity,
                        Scheduler::current()
                    ),
                    Err(NoReturn) => panic!("{:?} is waiting, but nothing was returned to it.  Bug likely in {:#?}", run_arc_process, Scheduler::current()),
                    Err(TooManyReturns(returns)) => panic!("{:?} got multiple returns: {:?}.  Stack is not being properly managed.", run_arc_process, returns)
                }
            }
        }
    }

    option_return_usize.unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Once;

    mod log_to_console {
        use super::*;

        #[test]
        fn with_1() {
            start_once();
            assert_eq!(log_to_console(1), 1);
        }

        #[test]
        fn with_2() {
            start_once();
            assert_eq!(log_to_console(2), 2);
        }

        #[test]
        fn with_4() {
            start_once();
            assert_eq!(log_to_console(4), 4);
        }

        #[test]
        fn with_8() {
            start_once();
            assert_eq!(log_to_console(8), 8);
        }

        #[test]
        fn with_16() {
            start_once();
            assert_eq!(log_to_console(16), 16);
        }

        #[test]
        fn with_32() {
            start_once();
            assert_eq!(log_to_console(32), 32);
        }

        #[test]
        fn with_64() {
            start_once();
            assert_eq!(log_to_console(64), 64);
        }

        #[test]
        fn with_128() {
            start_once();
            assert_eq!(log_to_console(128), 128);
        }

        #[test]
        fn with_256() {
            start_once();
            assert_eq!(log_to_console(256), 256);
        }

        #[test]
        fn with_512() {
            start_once();
            assert_eq!(log_to_console(512), 512);
        }

        #[test]
        fn with_1024() {
            start_once();
            assert_eq!(log_to_console(1024), 1024);
        }

        #[test]
        fn with_2048() {
            start_once();
            assert_eq!(log_to_console(2048), 2048);
        }

        #[test]
        fn with_4096() {
            start_once();
            assert_eq!(log_to_console(4096), 4096);
        }

        #[test]
        fn with_8192() {
            start_once();
            assert_eq!(log_to_console(8192), 8192);
        }

        #[test]
        fn with_16384() {
            start_once();
            assert_eq!(log_to_console(16384), 16384);
        }

        #[test]
        fn with_32768() {
            start_once();
            assert_eq!(log_to_console(32768), 32768);
        }

        #[test]
        fn with_65536() {
            start_once();
            assert_eq!(log_to_console(65536), 65536);
        }

        #[test]
        fn with_131072() {
            start_once();
            assert_eq!(log_to_console(131072), 131072);
        }
    }

    mod log_to_dom {
        use super::*;

        #[test]
        fn with_1() {
            start_once();
            assert_eq!(log_to_dom(1), 1);
        }

        #[test]
        fn with_2() {
            start_once();
            assert_eq!(log_to_dom(2), 2);
        }

        #[test]
        fn with_4() {
            start_once();
            assert_eq!(log_to_dom(4), 4);
        }

        #[test]
        fn with_8() {
            start_once();
            assert_eq!(log_to_dom(8), 8);
        }

        #[test]
        fn with_16() {
            start_once();
            assert_eq!(log_to_dom(16), 16);
        }

        #[test]
        fn with_32() {
            start_once();
            assert_eq!(log_to_dom(32), 32);
        }

        #[test]
        fn with_64() {
            start_once();
            assert_eq!(log_to_dom(64), 64);
        }

        #[test]
        fn with_128() {
            start_once();
            assert_eq!(log_to_dom(128), 128);
        }

        #[test]
        fn with_256() {
            start_once();
            assert_eq!(log_to_dom(256), 256);
        }

        #[test]
        fn with_512() {
            start_once();
            assert_eq!(log_to_dom(512), 512);
        }

        #[test]
        fn with_1024() {
            start_once();
            assert_eq!(log_to_dom(1024), 1024);
        }

        #[test]
        fn with_2048() {
            start_once();
            assert_eq!(log_to_dom(2048), 2048);
        }

        #[test]
        fn with_4096() {
            start_once();
            assert_eq!(log_to_dom(4096), 4096);
        }

        #[test]
        fn with_8192() {
            start_once();
            assert_eq!(log_to_dom(8192), 8192);
        }

        #[test]
        fn with_16384() {
            start_once();
            assert_eq!(log_to_dom(16384), 16384);
        }

        #[test]
        fn with_32768() {
            start_once();
            assert_eq!(log_to_dom(32768), 32768);
        }

        #[test]
        fn with_65536() {
            start_once();
            assert_eq!(log_to_dom(65536), 65536);
        }

        #[test]
        fn with_131072() {
            start_once();
            assert_eq!(log_to_dom(131072), 131072);
        }
    }

    static START: Once = Once::new();

    fn start_once() {
        START.call_once(|| {
            start();
        })
    }
}
