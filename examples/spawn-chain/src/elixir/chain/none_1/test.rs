use super::*;

use std::sync::Once;

use lumen_rt_core::registry;

use lumen_rt_full::process;
use lumen_rt_full::process::spawn::options::Options;
use lumen_rt_full::process::spawn::Spawned;
use lumen_rt_full::scheduler;

use crate::start::export_code;

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

#[test]
fn with_32768() {
    run_through(32768)
}

#[test]
fn with_65536() {
    run_through(65536)
}

fn inspect_code(arc_process: &Arc<Process>) -> frames::Result {
    let time_value = arc_process.stack_peek(1).unwrap();

    lumen_rt_full::sys::io::puts(&format!("{}", time_value));
    arc_process.remove_last_frame(1);

    Process::call_native_or_yield(arc_process)
}

fn run_through(n: usize) {
    start_once();

    let parent_process = None;
    let mut options: Options = Default::default();
    options.min_heap_size = Some(100 + 5 * n);
    let Spawned { process, .. } = process::spawn::code(
        parent_process,
        options,
        Atom::try_from_str("Elixir.ChainTest").unwrap(),
        Atom::try_from_str("inspect").unwrap(),
        &[],
        inspect_code,
    )
    .unwrap();
    super::place_frame_with_arguments(&process, Placement::Push, process.integer(n).unwrap())
        .unwrap();

    let arc_scheduler = scheduler::current();
    let arc_process = arc_scheduler.schedule(process);
    registry::put_pid_to_process(&arc_process);

    while scheduler::run_through(&arc_process) {}
}

static START: Once = Once::new();

fn start() {
    export_code();
}

fn start_once() {
    scheduler::set_unregistered_once();

    START.call_once(|| {
        start();
    })
}
