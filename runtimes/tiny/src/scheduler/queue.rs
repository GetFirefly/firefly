use std::collections::VecDeque;
use std::mem;
use std::sync::Arc;

use super::SchedulerData;

/// Just about the simplest of run queues, but it makes an attempt to ensure
/// that previously scheduled processes aren't starved by a continual
/// stream of new processes
#[derive(Default)]
pub(super) struct RunQueue {
    scheduled: VecDeque<Arc<SchedulerData>>,
    visited: VecDeque<Arc<SchedulerData>>,
}
impl RunQueue {
    /// Returns the next process to execute, if any are available
    pub fn next(&mut self) -> Option<Arc<SchedulerData>> {
        let next = self.scheduled.pop_front();
        if next.is_some() {
            return next;
        }
        // We've scheduled all processes at least once this cycle, start a new cycle,
        // but only if we've scheduled at least one process
        if self.visited.is_empty() {
            return None;
        }
        // To start a new cycle, we simply swap the empty schedule queue for
        // the visited queue and recurse
        mem::swap(&mut self.scheduled, &mut self.visited);
        self.scheduled.pop_front()
    }

    /// Schedules the given process immediately
    pub fn schedule_now(&mut self, process: Arc<SchedulerData>) {
        self.scheduled.push_front(process);
    }

    /// Schedules the given process for the first time, taking priority
    /// over previously scheduled processes which have already had an opportunity
    /// to execute this cycle
    ///
    /// In the most pathological of scenarios (an infinite spawn chain), one could
    /// starve older processes of run time, but that isn't something we're worried
    /// about here.
    pub fn schedule(&mut self, process: Arc<SchedulerData>) {
        self.scheduled.push_back(process)
    }

    /// Schedules the given process again after having just executed. All
    /// processes which have not executed this cycle will get to execute before
    /// this process runs again
    pub fn reschedule(&mut self, process: Arc<SchedulerData>) {
        self.visited.push_back(process);
    }
}
