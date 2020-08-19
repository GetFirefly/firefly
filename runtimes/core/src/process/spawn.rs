mod apply_3;
pub mod options;

use std::convert::TryInto;
use std::sync::Arc;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::exception::AllocResult;
use liblumen_alloc::erts::process::{Frame, FrameWithArguments, Native, Process};
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::{Arity, ModuleFunctionArity};

use crate::registry::put_pid_to_process;
use crate::scheduler::{self, Scheduled, Scheduler};

pub use self::options::{Connection, Options};

extern "Rust" {
    /// Makes `process` runnable in a target-specific manner, such as by putting stack frames for
    /// dealing with when the process runs out of code
    #[link_name = "lumen_rt_process_runnable"]
    fn runnable<'a>(
        process: &Process,
        frames_with_arguments_fn: Box<
            dyn FnOnce(&Process) -> AllocResult<Vec<FrameWithArguments>> + 'a,
        >,
    ) -> AllocResult<()>;
}

/// Spawns a process with arguments for `apply(module, function, arguments)` on its stack.
///
/// This allows the `apply/3` code to be changed with `apply_3::set_code(code)` to handle new
/// MFA unique to a given application.
pub fn apply_3(
    parent_process: &Process,
    options: Options,
    module: Atom,
    function: Atom,
    arguments: Term,
) -> exception::Result<Spawned> {
    let arity = arity(arguments);

    spawn(
        Some(parent_process),
        options,
        module,
        function,
        arity,
        Box::new(move |_| {
            let frame = apply_3::frame();
            let module_term = module.encode().unwrap();
            let function_term = function.encode().unwrap();
            let frame_with_arguments =
                frame.with_arguments(false, &[module_term, function_term, arguments]);

            Ok(vec![frame_with_arguments])
        }),
    )
}

/// Spawns a process with `arguments` on its stack and `native` run with those arguments instead
/// of passing through `apply/3`.
pub fn native<'a>(
    parent_process: Option<&Process>,
    options: Options,
    module: Atom,
    function: Atom,
    arguments: &'a [Term],
    native: Native,
) -> exception::Result<Spawned> {
    let arity = arguments.len() as u8;

    spawn(
        parent_process,
        options,
        module,
        function,
        arity,
        Box::new(move |_| {
            let module_function_arity = ModuleFunctionArity {
                module,
                function,
                arity,
            };
            let frame = Frame::new(module_function_arity, native);
            let frame_with_arguments = frame.with_arguments(false, arguments);

            Ok(vec![frame_with_arguments])
        }),
    )
}

pub fn spawn<'a>(
    parent_process: Option<&Process>,
    options: Options,
    module: Atom,
    function: Atom,
    arity: Arity,
    frames_with_arguments_fn: Box<
        dyn FnOnce(&Process) -> AllocResult<Vec<FrameWithArguments>> + 'a,
    >,
) -> exception::Result<Spawned> {
    let child_process = options.spawn(parent_process, module, function, arity)?;

    unsafe { runnable(&child_process, frames_with_arguments_fn) }?;

    // Connect after placing frame, so that any logging can show the `Frame`s when connections occur
    let connection = options.connect(parent_process, &child_process);

    Ok(Spawned {
        process: child_process,
        connection,
    })
}

pub struct Spawned {
    pub process: Process,
    #[must_use]
    pub connection: Connection,
}

impl Spawned {
    pub fn schedule_with_parent(self, parent: &Process) -> scheduler::Spawned {
        self.schedule_with_scheduler(parent.scheduler().unwrap())
    }

    pub fn schedule_with_scheduler(self, scheduler: Arc<dyn Scheduler>) -> scheduler::Spawned {
        let Self {
            process,
            connection,
        } = self;
        let arc_process = scheduler.schedule(process);

        put_pid_to_process(&arc_process);

        scheduler::Spawned {
            arc_process,
            connection,
        }
    }
}

// Private

fn arity(arguments: Term) -> u8 {
    match arguments.decode().unwrap() {
        TypedTerm::Nil => 0,
        TypedTerm::List(cons) => cons.count().unwrap().try_into().unwrap(),
        _ => {
            panic!(
                "Arguments {:?} are neither an empty nor a proper list",
                arguments
            );
        }
    }
}
