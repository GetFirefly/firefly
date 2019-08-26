//#![deny(warnings)]

use std::sync::Arc;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::{ProcessControlBlock, Status};
use liblumen_alloc::erts::term::{atom_unchecked, Atom, Term};
use liblumen_alloc::erts::ModuleFunctionArity;

use lumen_runtime::process::spawn::options::Options;
use lumen_runtime::scheduler::Scheduler;
use lumen_runtime::system;

pub mod code;
mod exec;
mod module;
pub use module::NativeModule;
mod native;
mod vm;
pub mod call_result;

#[cfg(test)]
mod tests;

use self::vm::VMState;
use lazy_static::lazy_static;

lazy_static! {
    pub static ref VM: VMState = VMState::new();
}

pub fn call_erlang(
    proc: Arc<ProcessControlBlock>,
    module: Atom,
    function: Atom,
    args: &[Term],
) -> std::result::Result<(), ()> {
    let return_ok = {
        let mfa = ModuleFunctionArity {
            module: Atom::try_from_str("lumen_eir_interpreter_intrinsics").unwrap(),
            function: Atom::try_from_str("return_ok").unwrap(),
            arity: 1,
        };
        proc.closure_with_env_from_slice(mfa.into(), crate::code::return_ok, proc.pid_term(), &[])
            .unwrap()
    };
    let return_throw = {
        let mfa = ModuleFunctionArity {
            module: Atom::try_from_str("lumen_eir_interpreter_intrinsics").unwrap(),
            function: Atom::try_from_str("return_throw").unwrap(),
            arity: 3,
        };
        proc.closure_with_env_from_slice(
            mfa.into(),
            crate::code::return_throw,
            proc.pid_term(),
            &[],
        )
        .unwrap()
    };

    let mut args_vec = vec![return_ok, return_throw];
    args_vec.extend(args.iter().cloned());

    let arguments = proc
        .list_from_slice(&args_vec)
    // if not enough memory here, resize `spawn_init` heap
        .unwrap();

    let options: Options = Default::default();
    //options.min_heap_size = Some(100_000);

    let run_arc_process = Scheduler::spawn_apply_3(
        &proc,
        options,
        module,
        function,
        arguments,
    )
    // if this fails increase heap size
    .unwrap();

    loop {
        let ran = Scheduler::current().run_through(&run_arc_process);

        match *run_arc_process.status.read() {
            Status::Exiting(ref exception) => match exception {
                exception::runtime::Exception {
                    class: exception::runtime::Class::Exit,
                    reason,
                    ..
                } => {
                    if *reason != atom_unchecked("normal") {
                        return Err(());
                    //panic!("ProcessControlBlock exited: {:?}", reason);
                    } else {
                        return Ok(());
                    }
                }
                _ => {
                    return Err(());
                    //panic!(
                    //    "ProcessControlBlock exception: {:?}\n{:?}",
                    //    exception,
                    //    run_arc_process.stacktrace()
                    //);
                }
            },
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
