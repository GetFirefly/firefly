use std::sync::Arc;

use liblumen_alloc::Process;

use crate::process::spawn::options::Options;
use crate::scheduler::{self, Spawned};

use super::r#loop;

pub fn default() -> Arc<Process> {
    child(&init())
}

pub fn init() -> Arc<Process> {
    super::once_crate();

    // During test allow multiple unregistered init processes because in tests, the `Scheduler`s
    // keep getting `Drop`ed as threads end.

    scheduler::current()
        .spawn_init(
            // init process being the parent process needs space for the arguments when spawning
            // child processes.  These will not be GC'd, so it can be a lot of space if proptest
            // needs to generate a lot of processes.
            16_000,
        )
        .unwrap()
}

pub fn child(parent_process: &Process) -> Arc<Process> {
    let mut options: Options = Default::default();
    options.min_heap_size = Some(16_000);
    let module = r#loop::module();
    let function = r#loop::function();
    let arguments = &[];
    let code = r#loop::code;

    let Spawned {
        arc_process: child_arc_process,
        connection,
    } = scheduler::spawn_code(parent_process, options, module, function, arguments, code).unwrap();
    assert!(!connection.linked);
    assert!(connection.monitor_reference.is_none());

    child_arc_process
}
