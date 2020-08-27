use std::sync::{Arc, Once};

use panic_control::chain_hook_ignoring;

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::runtime::process::set_log_exit;
use crate::runtime::process::spawn::Options;
use crate::runtime::scheduler::{self, Scheduled, Spawned};
use crate::{erlang, runtime};

use super::loop_0;

pub fn default() -> Arc<Process> {
    child(&init())
}

pub fn init() -> Arc<Process> {
    runtime::test::once(&[
        erlang::apply_3::function_symbol(),
        erlang::exit_1::function_symbol(),
        erlang::number_or_badarith_1::function_symbol(),
        erlang::self_0::function_symbol(),
        super::anonymous_0::function_symbol(),
        super::anonymous_1::function_symbol(),
        super::init::start_0::function_symbol(),
        loop_0::function_symbol(),
    ]);

    set_log_exit(false);

    ONCE.call_once(|| {
        // Ignore panics created by full runtime's `__lumen_start_panic`.  `catch_unwind` although
        // it stops the panic does not suppress the printing of the panic message and stack
        // backtrace without this.
        chain_hook_ignoring::<Term>();
    });

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
    let module = loop_0::module();
    let function = loop_0::function();
    let arguments = vec![];
    let mut options: Options = Default::default();
    options.min_heap_size = Some(16_000);

    let Spawned {
        arc_process: child_arc_process,
        connection,
    } = parent_process
        .scheduler()
        .unwrap()
        .spawn_module_function_arguments(Some(parent_process), module, function, arguments, options)
        .unwrap();

    assert!(!connection.linked);
    assert!(connection.monitor_reference.is_none());

    child_arc_process
}

static ONCE: Once = Once::new();
