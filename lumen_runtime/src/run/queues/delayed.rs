use std::collections::vec_deque::VecDeque;
use std::sync::Arc;

use crate::process::Process;
use crate::run::Run;
use crate::scheduler::Priority;

/// A run queue where the `Arc<Process` is run only when its delay is `0`.  This allows
/// the `Priority::Normal` and `Priority::Low` to be in same `Delayed` run queue, but for the
/// `Priority::Normal` to be run more often.
#[derive(Debug, Default)]
pub struct Delayed(VecDeque<DelayedProcess>);

impl Delayed {
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
