mod inspect;

use std::sync::Once;

use liblumen_alloc::erts::term::prelude::*;

use lumen_rt_core::registry;

use lumen_rt_full::process;
use lumen_rt_full::process::spawn::options::Options;
use lumen_rt_full::process::spawn::Spawned;
use lumen_rt_full::scheduler;

use crate::start::initialize_dispatch_table;

#[test]
fn with_1() {
    run_through(1)
}

#[test]
fn with_2() {
    run_through(2)
}

#[test]
fn with_4() {
    run_through(4)
}

#[test]
fn with_8() {
    run_through(8)
}

#[test]
fn with_16() {
    run_through(16)
}

#[test]
fn with_32() {
    run_through(32)
}

#[test]
fn with_64() {
    run_through(64)
}

#[test]
fn with_128() {
    run_through(128)
}

#[test]
fn with_256() {
    run_through(256)
}

#[test]
fn with_512() {
    run_through(512)
}

#[test]
fn with_1024() {
    run_through(1024)
}

#[test]
fn with_2048() {
    run_through(2048)
}

#[test]
fn with_4096() {
    run_through(4096)
}

#[test]
fn with_8192() {
    run_through(8192)
}

#[test]
fn with_16384() {
    run_through(16384)
}

fn module() -> Atom {
    Atom::from_str("Elixir.ChainTest")
}

#[allow(dead_code)]
fn module_id() -> usize {
    module().id()
}

fn run_through(n: usize) {
    start_once();

    let parent_process = None;
    let mut options: Options = Default::default();
    options.min_heap_size = Some(100 + 5 * n);
    let Spawned { process, .. } = process::spawn::spawn(
        parent_process,
        options,
        module(),
        inspect::function(),
        inspect::ARITY,
        Box::new(|child_process| {
            let n_term = child_process.integer(n)?;

            Ok(vec![
                super::frame().with_arguments(false, &[n_term]),
                inspect::frame().with_arguments(true, &[]),
            ])
        }),
    )
    .unwrap();

    let arc_scheduler = scheduler::current();
    let arc_process = arc_scheduler.schedule(process);
    registry::put_pid_to_process(&arc_process);

    while !arc_process.is_exiting() && scheduler::run_through(&arc_process) {}
}

static START: Once = Once::new();

fn start() {
    initialize_dispatch_table();
}

fn start_once() {
    START.call_once(|| {
        start();
    })
}
