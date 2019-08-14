use std::rc::Rc;
use std::sync::RwLock;

use libeir_ir::FunctionIdent;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::{heap, next_heap_size, Status};
use liblumen_alloc::erts::term::{atom_unchecked, Atom, Term};
use lumen_runtime::scheduler::Scheduler;
use lumen_runtime::system;

use super::module::ModuleRegistry;

pub struct VMState {
    pub modules: RwLock<ModuleRegistry>,
    pub closure_hack: RwLock<Vec<Vec<Term>>>,
}

impl VMState {
    pub fn new() -> Self {
        lumen_runtime::otp::erlang::apply_3::set_code(crate::code::apply);

        let mut modules = ModuleRegistry::new();
        modules.register_native_module(crate::native::make_erlang());

        VMState {
            modules: RwLock::new(modules),
            closure_hack: RwLock::new(Vec::new()),
        }
    }

    pub fn call(
        &mut self,
        fun: &FunctionIdent,
        args: &[Term],
    ) -> Result<Rc<Term>, (Rc<Term>, Rc<Term>, Rc<Term>)> {
        let arc_scheduler = Scheduler::current();
        let init_arc_process = arc_scheduler.spawn_init(0).unwrap();

        let module = Atom::try_from_str(&fun.module.as_str()).unwrap();
        let function = Atom::try_from_str(&fun.name.as_str()).unwrap();
        let arguments = init_arc_process
            .list_from_slice(args)
            // if not enough memory here, resize `spawn_init` heap
            .unwrap();

        let heap_size = next_heap_size(4 + 1000 * 2);
        // if this fails the entire tab is out-of-memory
        let heap = heap(heap_size).unwrap();

        let run_arc_process = Scheduler::spawn_apply_3(
            &init_arc_process,
            module,
            function,
            arguments,
            heap,
            heap_size)
            // if this fails  a bigger sized heap
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
                            panic!("ProcessControlBlock exited: {:?}", reason);
                        } else {
                            panic!("yay!");
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
}
