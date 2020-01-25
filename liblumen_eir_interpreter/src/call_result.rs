mod return_ok;
mod return_throw;

use core::ptr::NonNull;

use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;

use anyhow::*;

use liblumen_alloc::erts::process::{Process, Status};
use liblumen_alloc::erts::term::closure::Definition;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::erts::HeapFragment;

use lumen_runtime::process::spawn::options::Options;
use lumen_runtime::scheduler::{Scheduler, Spawned};
use lumen_runtime::system;

/// A sort of ghetto-future used to get the result from a process
/// spawn.
pub struct ProcessResultReceiver {
    pub process: Arc<Process>,
    rx: Receiver<ProcessResult>,
}

impl ProcessResultReceiver {
    pub fn try_get(&self) -> Option<ProcessResult> {
        self.rx.try_recv().ok()
    }
}

pub struct ProcessResult {
    pub heap: NonNull<HeapFragment>,
    pub result: Result<Term, (Term, Term, Term)>,
}

struct ProcessResultSender {
    tx: Sender<ProcessResult>,
}

pub fn call_run_erlang(
    proc: Arc<Process>,
    module: Atom,
    function: Atom,
    args: &[Term],
) -> ProcessResult {
    let recv = call_erlang(proc, module, function, args);
    let run_arc_process = recv.process.clone();

    loop {
        let ran = Scheduler::current().run_through(&run_arc_process);

        match *run_arc_process.status.read() {
            Status::Exiting(_) => {
                return recv.try_get().unwrap();
            }
            Status::Waiting => {
                if ran {
                    system::io::puts(&format!(
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
                system::io::puts(&format!(
                    "RUNNABLE Run queues len = {:?}",
                    Scheduler::current().run_queues_len()
                ));
            }
            Status::Running => {
                system::io::puts(&format!(
                    "RUNNING Run queues len = {:?}",
                    Scheduler::current().run_queues_len()
                ));
            }
        }
    }
}

pub fn call_erlang(
    proc: Arc<Process>,
    module: Atom,
    function: Atom,
    args: &[Term],
) -> ProcessResultReceiver {
    let (tx, rx) = channel();

    let sender = ProcessResultSender { tx };
    let sender_term = proc.resource(Box::new(sender)).unwrap();

    let return_ok = {
        let module = Atom::try_from_str("lumen_eir_interpreter_intrinsics").unwrap();
        const ARITY: u8 = 1;
        let definition = Definition::Anonymous {
            // TODO assign `index` scoped to `module`
            index: 0,
            // TODO calculate `old_unique` for `return_ok` with `sender_term` captured.
            old_unique: Default::default(),
            // TODO calculate `unique` for `return_ok` with `sender_term` captured.
            unique: Default::default(),
            creator: proc.pid().into(),
        };

        proc.closure_with_env_from_slice(
            module,
            definition,
            ARITY,
            Some(return_ok::LOCATED_CODE),
            &[sender_term],
        )
        .unwrap()
    };

    let return_throw = {
        let module = Atom::try_from_str("lumen_eir_interpreter_intrinsics").unwrap();
        let definition = Definition::Anonymous {
            // TODO assign `index` scoped to `module`
            index: 1,
            // TODO calculate `old_unique` for `return_throw` with `sender_term` captured.
            old_unique: Default::default(),
            // TODO calculate `unique` for `return_throw` with `sender_term` captured.
            unique: Default::default(),
            creator: proc.pid().into(),
        };
        const ARITY: u8 = 1;

        proc.closure_with_env_from_slice(
            module,
            definition,
            ARITY,
            Some(return_throw::LOCATED_CODE),
            &[sender_term],
        )
        .unwrap()
    };

    let mut args_vec = vec![return_ok, return_throw];
    args_vec.extend(args.iter().cloned());

    let arguments = proc.list_from_slice(&args_vec).unwrap();

    let options: Options = Default::default();
    //options.min_heap_size = Some(100_000);

    let Spawned {
        arc_process: run_arc_process,
        ..
    } = Scheduler::spawn_apply_3(&proc, options, module, function, arguments).unwrap();

    ProcessResultReceiver {
        process: run_arc_process,
        rx,
    }
}
