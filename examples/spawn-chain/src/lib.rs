#![deny(warnings)]
#![feature(type_ascription)]

use std::sync::Arc;

use lumen_runtime::atom::Existence::DoNotCare;
use lumen_runtime::exception::{self, Exception};
use lumen_runtime::process::{IntoProcess, Process, Status};
use lumen_runtime::registry;
use lumen_runtime::scheduler::Scheduler;
use lumen_runtime::term::Term;

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
                        panic!("Process exited: {:?}", reason)
                    } else {
                        break;
                    }
                }
                _ => {
                    crate::code::print_stacktrace(&run_arc_process);
                    panic!("Process exception: {:?}", exception)
                }
            },
            Status::Waiting => {
                web_sys::console::log_1(
                    &format!(
                        "WAITING Run queues len = {:?}",
                        Scheduler::current().run_queues_len()
                    )
                    .into(),
                );
            }
            Status::Runnable => {
                web_sys::console::log_1(
                    &format!(
                        "RUNNABLE Run queues len = {:?}",
                        Scheduler::current().run_queues_len()
                    )
                    .into(),
                );
            }
            Status::Running => {
                web_sys::console::log_1(
                    &format!(
                        "RUNNING Run queues len = {:?}",
                        Scheduler::current().run_queues_len()
                    )
                    .into(),
                );
            }
        }
    }
}
