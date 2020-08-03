use std::borrow::Borrow;
use std::collections::vec_deque::VecDeque;
use std::collections::HashSet;
use std::fmt::{self, Debug};
use std::hash::Hash;
use std::sync::Arc;

use liblumen_alloc::erts::process::{Priority, Process, Status};

use crate::scheduler::Run;

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
            Next::Wait => {
                self.waiting.insert(arc_process);
                None
            }
            Next::PushBack => {
                self.enqueue(arc_process);
                None
            }
            Next::Exit => Some(arc_process),
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
            Status::Runnable => Next::PushBack,
            Status::Waiting => Next::Wait,
            Status::RuntimeException(_) => Next::Exit,
            Status::SystemException(_) => {
                unreachable!("System exception should have already been cleared")
            }
            Status::Running => {
                unreachable!("Process.stop_running() should have been called before this")
            }
            Status::Unrunnable => {
                unreachable!("runtime::process::runnable(process) show have been called before attempting to run a process with a scheduler")
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

/// A run queue where the `Arc<Process>` is run immediately when it is encountered
#[derive(Debug, Default)]
pub struct Immediate(VecDeque<Arc<Process>>);

impl Immediate {
    pub fn contains(&self, value: &Arc<Process>) -> bool {
        self.0.contains(value)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn dequeue(&mut self) -> Run {
        match self.0.pop_front() {
            Some(arc_process) => Run::Now(arc_process),
            None => Run::None,
        }
    }

    pub fn enqueue(&mut self, process: Arc<Process>) {
        self.0.push_back(process);
    }
}

/// A run queue where the `Arc<Process` is run only when its delay is `0`.  This allows
/// the `Priority::Normal` and `Priority::Low` to be in same `Delayed` run queue, but for the
/// `Priority::Normal` to be run more often.
#[derive(Debug, Default)]
pub struct Delayed(VecDeque<DelayedProcess>);

impl Delayed {
    pub fn contains(&self, value: &Arc<Process>) -> bool {
        self.0
            .iter()
            .any(|delayed_process| &delayed_process.arc_process == value)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn dequeue(&mut self) -> Run {
        match self.0.pop_front() {
            Some(mut delayed_process) => {
                if delayed_process.delay == 0 {
                    Run::Now(delayed_process.arc_process)
                } else {
                    delayed_process.delay -= 1;
                    self.0.push_back(delayed_process);

                    Run::Delayed
                }
            }
            None => Run::None,
        }
    }

    pub fn enqueue(&mut self, arc_process: Arc<Process>) {
        let delayed_process = DelayedProcess::new(arc_process);
        self.0.push_back(delayed_process);
    }
}

type Delay = u8;

#[derive(Debug)]
struct DelayedProcess {
    delay: Delay,
    arc_process: Arc<Process>,
}

impl DelayedProcess {
    fn new(arc_process: Arc<Process>) -> DelayedProcess {
        DelayedProcess {
            delay: Self::priority_to_delay(arc_process.priority),
            arc_process,
        }
    }

    fn priority_to_delay(priority: Priority) -> Delay {
        // BEAM can use pre-decrement (`--p->schedule_count`), but we can't in Rust, so use `delay`
        // instead of `schedule_count` and decrement only if `Priority::Low`.
        match priority {
            Priority::Low => 7,
            Priority::Normal => 0,
            _ => unreachable!(),
        }
    }
}
