use std::convert::TryInto;
use std::sync::Arc;

use liblumen_alloc::erts::exception::AllocResult;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::{
    exception, Arity, Frame, FrameWithArguments, ModuleFunctionArity, Native, Process,
};

use liblumen_core::locks::Mutex;
pub use lumen_rt_core::future::*;
use lumen_rt_core::process::current_process;
use lumen_rt_core::process::spawn::Options;
use lumen_rt_core::registry;

use crate::process::spawn;
use crate::scheduler;

pub fn run_until_ready<F>(
    options: Options,
    frames_with_arguments_fn: F,
    max_scheduler_runs: usize,
) -> Result<Ready, NotReady>
where
    F: FnOnce(&Process) -> AllocResult<Vec<FrameWithArguments>>,
{
    assert!(!options.link);
    assert!(!options.monitor);

    let spawned = spawn(options, frames_with_arguments_fn)?;

    spawned.run_until_ready(max_scheduler_runs)
}

// Private

const ARITY: Arity = 2;

fn frame() -> Frame {
    Frame::new(module_function_arity(), Native::Two(native))
}

fn function() -> Atom {
    Atom::from_str("future")
}

fn module() -> Atom {
    Atom::from_str("Elixir.Lumen")
}

fn module_function_arity() -> ModuleFunctionArity {
    ModuleFunctionArity {
        module: module(),
        function: function(),
        arity: ARITY,
    }
}

pub extern "C" fn native(value: Term, future: Term) -> Term {
    let future_resource_box: Boxed<Resource> = future.try_into().unwrap();
    let future_resource: Resource = future_resource_box.into();
    let future_mutex: &Arc<Mutex<Future>> = future_resource.downcast_ref().unwrap();

    future_mutex.lock().ready(Ready {
        arc_process: current_process().clone(),
        result: Ok(value),
    });

    value
}

fn spawn<F>(options: Options, frames_with_arguments_fn: F) -> exception::Result<Spawned>
where
    F: FnOnce(&Process) -> AllocResult<Vec<FrameWithArguments>>,
{
    let parent_process = None;
    let arc_mutex_future: Arc<Mutex<Future>> = Default::default();

    let spawn::Spawned { process, .. } = spawn::spawn(
        parent_process,
        options,
        module(),
        function(),
        ARITY,
        |child_process: &Process| {
            let mut vec: Vec<FrameWithArguments> = Vec::new();

            let mut frames_with_arguments = frames_with_arguments_fn(child_process)?;
            vec.append(&mut frames_with_arguments);

            let future_resource = child_process.resource(Box::new(arc_mutex_future.clone()))?;
            vec.push(frame().with_arguments(true, &[future_resource]));

            Ok(vec)
        },
    )?;

    let arc_process = scheduler::current().schedule(process);
    registry::put_pid_to_process(&arc_process);

    Ok(Spawned {
        arc_process,
        arc_mutex_future,
    })
}
