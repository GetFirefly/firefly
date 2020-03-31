#[cfg(feature = "runtime_minimal")]
use std::ffi::c_void;
#[cfg(feature = "runtime_minimal")]
use std::mem;
use std::sync::Arc;

#[cfg(feature = "runtime_minimal")]
use liblumen_core::symbols::FunctionSymbol;

#[cfg(feature = "runtime_minimal")]
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::Process;
#[cfg(feature = "runtime_minimal")]
use liblumen_core::sys::dynamic_call::DynamicCallee;

use crate::runtime;
use crate::runtime::process::spawn::options::Options;
use crate::runtime::scheduler::{self, Spawned};

use super::r#loop;

pub fn default() -> Arc<Process> {
    child(&init())
}

pub fn init() -> Arc<Process> {
    #[cfg(feature = "runtime_minimal")]
    runtime::test::once(&[FunctionSymbol {
        module: Atom::from_str("init").id(),
        function: Atom::from_str("start").id(),
        arity: 0,
        ptr: unsafe { mem::transmute::<DynamicCallee, *const c_void>(init_start) },
    }]);
    #[cfg(feature = "runtime_full")]
    runtime::test::once(&[]);
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

#[cfg(feature = "runtime_minimal")]
extern "C" {
    #[link_name = "__lumen_builtin_yield"]
    fn builtin_yield() -> bool;
}

#[cfg(feature = "runtime_minimal")]
extern "C" fn init_start() -> usize {
    loop {
        lumen_rt_minimal::process::current_process().wait();

        unsafe {
            builtin_yield();
        }
    }
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
