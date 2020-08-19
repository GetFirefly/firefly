use std::convert::TryInto;
use std::sync::Arc;

use liblumen_core::locks::Mutex;

use liblumen_alloc::erts::exception::{self, AllocResult, Exception};
use liblumen_alloc::erts::process::{Frame, FrameWithArguments, Native, Process, Status};
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::{Arity, ModuleFunctionArity};

use crate::process::current_process;
use crate::process::spawn::{self, Options};
use crate::{registry, scheduler};

pub fn run_until_ready(
    options: Options,
    frames_with_arguments_fn: Box<dyn FnOnce(&Process) -> AllocResult<Vec<FrameWithArguments>>>,
    max_scheduler_runs: usize,
) -> Result<Ready, NotReady> {
    assert!(!options.link);
    assert!(!options.monitor);

    let spawned = spawn(options, frames_with_arguments_fn)?;

    spawned.run_until_ready(max_scheduler_runs)
}

pub struct Ready {
    pub arc_process: Arc<Process>,
    pub result: exception::Result<Term>,
}

impl Clone for Ready {
    fn clone(&self) -> Self {
        Ready {
            arc_process: self.arc_process.clone(),
            result: match &self.result {
                Ok(term) => Ok(*term),
                Err(exception) => Err(exception.clone()),
            },
        }
    }
}

#[derive(Debug)]
pub enum NotReady {
    RunLimit { runs: usize },
    Failed(Exception),
}

impl From<Exception> for NotReady {
    fn from(err: Exception) -> Self {
        NotReady::Failed(err)
    }
}

pub enum Future {
    /// Future was spawned
    Spawned,
    /// Future's process completed.
    ///
    /// If `result`:
    /// * `Ok(Term)` - the `process` completed and `Term` is the return given to `code`.
    /// * `Err(exception::Exception)` - the `process` exited with an exception.
    Ready(Ready),
}

impl Future {
    pub fn ready(&mut self, ready: Ready) {
        *self = Future::Ready(ready);
    }
}

impl Default for Future {
    fn default() -> Self {
        Self::Spawned
    }
}

pub struct Spawned {
    pub arc_process: Arc<Process>,
    pub arc_mutex_future: Arc<Mutex<Future>>,
}

impl Spawned {
    pub fn run_until_ready(&self, max_scheduler_runs: usize) -> Result<Ready, NotReady> {
        let scheduler = scheduler::current();

        for _ in 0..max_scheduler_runs {
            assert!(scheduler.run_once());

            if let Future::Ready(ref ready) = *self.arc_mutex_future.lock() {
                return Ok(ready.clone());
            }

            if let Status::RuntimeException(ref exception) = *self.arc_process.status.read() {
                return Ok(Ready {
                    arc_process: self.arc_process.clone(),
                    result: Err(exception::Exception::Runtime(exception.clone())),
                });
            }
        }

        Err(NotReady::RunLimit {
            runs: max_scheduler_runs,
        })
    }
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

fn spawn(
    options: Options,
    frames_with_arguments_fn: Box<dyn FnOnce(&Process) -> AllocResult<Vec<FrameWithArguments>>>,
) -> exception::Result<Spawned> {
    let parent_process = None;
    let arc_mutex_future: Arc<Mutex<Future>> = Default::default();
    let child_arc_mutex_future = arc_mutex_future.clone();

    let spawn::Spawned { process, .. } = spawn::spawn(
        parent_process,
        options,
        module(),
        function(),
        ARITY,
        Box::new(|child_process: &Process| {
            let mut vec: Vec<FrameWithArguments> = Vec::new();

            let mut frames_with_arguments = frames_with_arguments_fn(child_process)?;
            vec.append(&mut frames_with_arguments);

            let future_resource = child_process.resource(child_arc_mutex_future);
            vec.push(frame().with_arguments(true, &[future_resource]));

            Ok(vec)
        }),
    )?;

    let arc_process = scheduler::current().schedule(process);
    registry::put_pid_to_process(&arc_process);

    Ok(Spawned {
        arc_process,
        arc_mutex_future,
    })
}
