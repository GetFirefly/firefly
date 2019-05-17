use std::collections::HashSet;
use std::sync::Arc;

use crate::process::{Process, Status};
use crate::run::queues::delayed::Delayed;
use crate::run::queues::immediate::Immediate;
use crate::run::queues::Next::*;
use crate::run::Run;
use crate::scheduler::Priority;

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
    #[cfg(test)]
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
        let next = Next::from_status(&arc_process.status.read().unwrap());

        // has to be separate so that `arc_process` can be moved
        match next {
            Wait => {
                self.waiting.insert(arc_process);
                None
            }
            PushBack => {
                self.enqueue(arc_process);
                None
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

type Waiting = HashSet<Arc<Process>>;
