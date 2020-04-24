use std::sync::Arc;

use liblumen_core::locks::Mutex;

use liblumen_alloc::erts::exception::{self, Exception};
use liblumen_alloc::erts::process::{Process, Status};
use liblumen_alloc::erts::term::prelude::*;

use crate::scheduler;

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
