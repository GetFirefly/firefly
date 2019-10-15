use std::convert::TryInto;
use std::sync::Arc;

use liblumen_core::locks::Mutex;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::exception::system::Alloc;
use liblumen_alloc::erts::process::code;
use liblumen_alloc::erts::process::{Process, Status};
use liblumen_alloc::erts::term::{resource, Atom};

use crate::process;
use crate::process::spawn;
use crate::process::spawn::options::{Connection, Options};
use crate::registry;
use crate::scheduler::Scheduler;
use crate::time::monotonic::{time_in_milliseconds, Milliseconds};

pub fn run_until_ready<PlaceFrameWithArguments>(
    options: Options,
    place_frame_with_arguments: PlaceFrameWithArguments,
    timeout: Milliseconds,
) -> Result<Ready, NotReady>
where
    PlaceFrameWithArguments: Fn(&Process) -> Result<(), Alloc>,
{
    assert!(!options.link);
    assert!(!options.monitor);

    let Spawned {
        arc_process,
        arc_mutex_future,
        connection,
    } = spawn(options, place_frame_with_arguments)?;

    assert!(!connection.linked);
    assert!(connection.monitor_reference.is_none());

    let scheduler = Scheduler::current();
    let end = time_in_milliseconds() + timeout;

    while time_in_milliseconds() < end {
        assert!(scheduler.run_once());

        if let Future::Ready(ref ready) = *arc_mutex_future.lock() {
            return Ok(ready.clone());
        }

        if let Status::Exiting(ref exception) = *arc_process.status.read() {
            return Ok(Ready {
                arc_process: arc_process.clone(),
                result: Err(exception::Exception::Runtime(exception.clone())),
            });
        }
    }

    Err(NotReady::Timeout { duration: timeout })
}

pub struct Ready {
    pub arc_process: Arc<Process>,
    pub result: exception::Result,
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
    Timeout { duration: Milliseconds },
    Alloc(Alloc),
}

impl From<Alloc> for NotReady {
    fn from(alloc: Alloc) -> Self {
        NotReady::Alloc(alloc)
    }
}

// Private

fn code(arc_process: &Arc<Process>) -> code::Result {
    let return_term = arc_process.stack_pop().unwrap();
    let future_term = arc_process.stack_pop().unwrap();
    assert!(future_term.is_resource_reference());

    let future_resource_reference: resource::Reference = future_term.try_into().unwrap();
    let future_mutex: &Arc<Mutex<Future>> = future_resource_reference.downcast_ref().unwrap();

    future_mutex.lock().ready(Ready {
        arc_process: arc_process.clone(),
        result: Ok(return_term),
    });

    arc_process.remove_last_frame();

    Process::call_code(arc_process)
}

fn function() -> Atom {
    Atom::try_from_str("future").unwrap()
}

fn module() -> Atom {
    Atom::try_from_str("Elixir.Lumen").unwrap()
}

fn spawn<PlaceFrameWithArguments>(
    options: Options,
    place_frame_with_arguments: PlaceFrameWithArguments,
) -> Result<Spawned, Alloc>
where
    PlaceFrameWithArguments: Fn(&Process) -> Result<(), Alloc>,
{
    let (
        spawn::Spawned {
            process,
            connection,
        },
        arc_mutex_future,
    ) = spawn_unscheduled(options)?;

    place_frame_with_arguments(&process)?;

    let arc_process = Scheduler::current().schedule(process);
    registry::put_pid_to_process(&arc_process);

    Ok(Spawned {
        arc_process,
        arc_mutex_future,
        connection,
    })
}

fn spawn_unscheduled(options: Options) -> Result<(spawn::Spawned, Arc<Mutex<Future>>), Alloc> {
    let parent_process = None;
    let arguments = &[];
    let spawned = process::spawn::code(
        parent_process,
        options,
        module(),
        function(),
        arguments,
        code,
    )?;

    let arc_mutex_future = Arc::new(Mutex::new(Default::default()));

    let process = &spawned.process;
    let future_resource_reference = process.resource(Box::new(arc_mutex_future.clone()))?;
    process.stack_push(future_resource_reference)?;

    Ok((spawned, arc_mutex_future))
}

enum Future {
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
    fn ready(&mut self, ready: Ready) {
        *self = Future::Ready(ready);
    }
}

impl Default for Future {
    fn default() -> Self {
        Self::Spawned
    }
}

struct Spawned {
    arc_process: Arc<Process>,
    arc_mutex_future: Arc<Mutex<Future>>,
    connection: Connection,
}
