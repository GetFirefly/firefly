use std::sync::Arc;

use liblumen_alloc::Process;

use crate::process::spawn::options::Options;
use crate::{process, scheduler};

use super::loop_0;

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
    let module = super::module();
    let function = loop_0::function();
    let arguments = &[];
    let native = loop_0::NATIVE;

    let spawned = process::spawn::native(
        Some(parent_process),
        options,
        module,
        function,
        arguments,
        native,
    )
    .unwrap();
    let connection = &spawned.connection;

    assert!(!connection.linked);
    assert!(connection.monitor_reference.is_none());

    let scheduled = spawned.schedule_with_parent(parent_process);

    scheduled.arc_process
}
