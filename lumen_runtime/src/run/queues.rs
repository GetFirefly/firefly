use core::borrow::Borrow;
use core::fmt::{self, Debug};
use core::hash::Hash;

use std::collections::HashSet;
use std::sync::Arc;

use anyhow::*;

use liblumen_alloc::erts::process::{Priority, Process, Status};

use crate::run::queues::delayed::Delayed;
use crate::run::queues::immediate::Immediate;
use crate::run::queues::Next::*;
use crate::run::Run;

mod delayed;
mod immediate;

#[derive(Debug, Default)]
pub struct Queues {
    waiting: Waiting,
    normal_low: Delayed,
    high: Immediate,
    max: Immediate,
}

impl Queues {
    pub fn contains(&self, value: &Arc<Process>) -> bool {
        self.waiting.contains(value)
            || self.normal_low.contains(value)
            || self.high.contains(value)
            || self.max.contains(value)
    }

    pub fn run_queue_len(&self, priority: Priority) -> usize {
        match priority {
            Priority::Low | Priority::Normal => self.normal_low.len(),
            Priority::High => self.high.len(),
            Priority::Max => self.max.len(),
        }
    }

    pub fn dequeue(&mut self) -> Run {
        if 0 < self.max.len() {
            self.max.dequeue()
        } else if 0 < self.high.len() {
            self.high.dequeue()
        } else if 0 < self.normal_low.len() {
            self.normal_low.dequeue()
        } else {
            Run::None
        }
    }

    pub fn enqueue(&mut self, arc_process: Arc<Process>) {
        match arc_process.priority {
            Priority::Low | Priority::Normal => self.normal_low.enqueue(arc_process),
            Priority::High => self.high.enqueue(arc_process),
            Priority::Max => self.max.enqueue(arc_process),
        }
    }

    pub fn len(&self) -> usize {
        self.waiting.len() + self.normal_low.len() + self.high.len() + self.max.len()
    }

    /// Returns the process is not pushed back because it is exiting
    #[must_use]
    pub fn requeue(&mut self, arc_process: Arc<Process>) -> Option<Arc<Process>> {
        let next = Next::from_status(&arc_process.status.read());

        // has to be separate so that `arc_process` can be moved
        match next {
            Wait => {
                self.waiting.insert(arc_process);
                None
            }
            PushBack => {
                if arc_process.code_stack_len() == 0 {
                    arc_process.exit_normal(anyhow!("Out of code").into());
                    Some(arc_process)
                } else {
                    self.enqueue(arc_process);
                    None
                }
            }
            Exit => Some(arc_process),
        }
    }

    pub fn stop_waiting(&mut self, process: &Process) {
        match self.waiting.get(process) {
            Some(arc_process) => {
                let arc_process = Arc::clone(arc_process);
                self.waiting.remove(&arc_process);

                self.enqueue(arc_process);
            }
            None => (),
        }
    }
}

// Private

enum Next {
    Wait,
    PushBack,
    Exit,
}

impl Next {
    fn from_status(status: &Status) -> Next {
        match status {
            Status::Runnable => PushBack,
            Status::Waiting => Wait,
            Status::Exiting(_) => Exit,
            Status::Running => {
                unreachable!("Process.stop_running() should have been called before this")
            }
        }
    }
}

#[derive(Default)]
pub struct Waiting(HashSet<Arc<Process>>);

impl Waiting {
    fn contains(&self, value: &Arc<Process>) -> bool {
        self.0.contains(value)
    }

    fn get<Q: ?Sized>(&self, value: &Q) -> Option<&Arc<Process>>
    where
        Arc<Process>: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.0.get(value)
    }

    fn insert(&mut self, waiter: Arc<Process>) -> bool {
        self.0.insert(waiter)
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn remove(&mut self, waiter: &Arc<Process>) -> bool {
        self.0.remove(waiter)
    }
}

impl Debug for Waiting {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut vec: Vec<_> = self.0.iter().collect();
        vec.sort_by_key(|arc_process| arc_process.pid());

        for arc_process in vec {
            write!(f, "{:?}:\n{:?}", arc_process, arc_process.stacktrace())?;
        }

        Ok(())
    }
}
