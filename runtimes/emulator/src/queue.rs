use std::cell::UnsafeCell;
use std::ops::Deref;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use crossbeam::deque::{Injector, Steal, Worker};

use firefly_rt::process::{Priority, Process, ProcessId};

/// A `Task` is work which can be scheduled in a `TaskQueue`
pub trait Task {
    type Id: Copy + PartialEq;

    /// The unique id of this task
    fn id(&self) -> Self::Id;

    /// The priority of this task
    fn priority(&self) -> Priority;

    /// This should return true if the task should be delayed/re-enqueued
    /// due to some criteria of the task not being met.
    ///
    /// This is used to implement multiple priorities within the same queue
    fn should_delay(&self) -> bool;
}
impl<T: Task> Task for Arc<T> {
    type Id = <T as Task>::Id;

    #[inline]
    fn id(&self) -> Self::Id {
        self.deref().id()
    }
    #[inline]
    fn priority(&self) -> Priority {
        self.deref().priority()
    }
    #[inline]
    fn should_delay(&self) -> bool {
        self.deref().should_delay()
    }
}
impl Task for Process {
    type Id = ProcessId;

    #[inline]
    fn id(&self) -> Self::Id {
        self.id()
    }

    #[inline]
    fn priority(&self) -> Priority {
        self.status(Ordering::Relaxed).priority()
    }

    #[inline]
    fn should_delay(&self) -> bool {
        // Delay a low-priority process up to 8 times
        let mut process = self.lock();
        process.schedule_count += 1;
        if process.schedule_count == 8 {
            process.schedule_count = 0;
            false
        } else {
            true
        }
    }
}

/// This trait defines the interface for a queue of `Task`
///
/// This trait allows us to define more specialized queues built
/// on simpler queues, without having to know anything about how they
/// are implemented.
pub trait TaskQueue {
    /// The type of task managed by this queue
    type Task: Task;

    /// Returns true if the queue is empty
    fn is_empty(&self) -> bool;
    /// Pushes a task at the end of the queue.
    ///
    /// In general this should allow all other tasks in the queue to be seen
    /// before `task`, but depending on the internal implementation, `task`
    /// might get scheduled earlier than others in the same queue.
    fn push(&self, task: Self::Task);
    /// Takes the next task from the queue
    fn pop(&self) -> Option<Self::Task>;
    /// Returns the number of tasks which have been scheduled from this queue
    /// during the current scheduling cycle.
    fn scheduled(&self) -> u8;
    /// This is called when starting a new scheduling cycle.
    ///
    /// The task queue should use this to reset any internal per-cycle statistics
    fn clear_statistics(&self);
}

/// The run queue is used to handle prioritization across processes, ports, and system tasks
///
/// Each type of task has its own set of queues, and potentially multiple priorities.
///
/// The scheduling algorithm here is vaguely similar to the interleaved weighted round-robin
/// algorithm, but is modified in a few ways:
///
/// * The weights are not fixed, but are adaptive based on what queues have tasks to be scheduled.
/// This is designed to preserve the proportion of scheduler time which each queue gets relative
/// to the others.
/// * We always serve higher priority tasks before lower priority tasks during a cycle
/// * Normal/low priority tasks are serviced from a single queue, and rather than using weights,
/// we artificially "delay" tasks of low priority until they have been enqueued a set number of
/// times without being scheduled, only scheduling them when that number is reached. This has a
/// slightly different effect than weighted queues, as it results in normal tasks getting serviced
/// before low priority tasks in almost all cases, except when a low priority task has been deferred
/// so long as to demand some execution time.
///
/// The flow goes something like this:
///
/// * A single scheduling "cycle" is composed of 10 "ticks" or iterations of the core loop.
/// * `max` tasks, when present, consume the first 5 ticks of a cycle, at which point lower
/// priority tasks are given time to execute, for the final 5 ticks of the cycle
/// * `hi` tasks, when present, consume the first 3 ticks of the 5 ticks given to tasks of
/// less than `max` priority. At which point `normal` and `low` tasks are given 2 ticks to execute.
/// * `normal` and `lo` tasks only get 2 ticks per cycle to execute when there are `max` and `hi`
/// tasks present, and they share the same queue. However, `lo` tasks which are set to be scheduled
/// during that tick are rescheduled until they have been scheduled 8 times, giving `normal` tasks
/// up to 8x more scheduler time than `lo` tasks when both are present.
/// * If there are no tasks in a particular priority queue, that queue is skipped and the execution
/// time it would have been allocated is redistributed between lower priority queues in such a way
/// as to preserve the proportionality of those individual queues to the total execution time.
///
/// This is a more complex and permissive priority system than ERTS, in which `max` and `hi` processes
/// starve all priorities lower than them on a single scheduler. Our model is designed to work in a
/// single-threaded environment where we may have `max` or `hi` priority tasks that we only wish to
/// give greater resources to, while avoiding starvation of lower priority tasks.
///
/// # Scaling Properties
///
/// Let's assume we have the following processes in the run queue:
///
/// * 10 `max`
/// * 10 `hi`
/// * 100 `normal`
/// * 10 `lo`
///
/// Let's further assume that we have 1000 "ticks" to schedule.
///
/// From the perspective of a process in each queue, here's what proportion of execution time
/// they get relative to all other processes in the various queues:
///
/// * `max`, 60 ticks per task (3x more than `hi`)
/// * `hi`, 20 ticks per task (10x more than `normal`)
/// * `normal`, 2 ticks per task (8x more than `lo`)
/// * `lo`, 1 tick every 4 cycles
///
/// Obviously these numbers change as the distribution of processes in different priorities changes.
/// In practice, most processes are `normal` or `lo`, while only a very limited few are `hi` or `max`,
/// with potentially none at all in `max`. When there are no `max` tasks present, the entire system has
/// more than double the throughput, because `max` tasks consume 6 ticks of a 10 tick cycle. As a result
/// careful consideration of process prioritization is important.
pub struct RunQueue<Q: TaskQueue> {
    /// This is the global queue for unscheduled tasks
    ///
    /// All newly spawned tasks go in this queue, to be picked up by the next available scheduler
    global: Arc<Injector<<Q as TaskQueue>::Task>>,
    /// This is a scheduler-local queue from which tasks can be stolen by other schedulers
    ///
    /// Tasks go in this queue when they are scheduled out by the scheduler, or when the scheduler
    /// grabs a batch of work from the global queue.
    ///
    /// This particular queue is for max priority tasks, but the other priorities work the same
    max: Q,
    /// Same as above, but for high priority tasks
    hi: Q,
    /// Same as above, but for normal priority tasks
    normal: Q,
}
impl<Q: TaskQueue + Default> RunQueue<Q> {
    pub fn new(global: Arc<Injector<<Q as TaskQueue>::Task>>) -> Self {
        Self {
            global,
            max: Q::default(),
            hi: Q::default(),
            normal: Q::default(),
        }
    }
}
impl<Q: TaskQueue> RunQueue<Q> {
    /// Steal tasks from the global queue into our local queues
    ///
    /// Returns `true` if there are tasks available after doing this.
    pub fn backfill(&self) -> bool {
        let inq = Worker::new_fifo();
        loop {
            match self.global.steal_batch(&inq) {
                Steal::Empty => return false,
                Steal::Retry => continue,
                Steal::Success(_) => break,
            }
        }

        while let Some(task) = inq.pop() {
            self.push(task);
        }

        true
    }
}
impl<Q: TaskQueue> TaskQueue for RunQueue<Q> {
    type Task = <Q as TaskQueue>::Task;

    fn is_empty(&self) -> bool {
        self.max.is_empty() && self.hi.is_empty() && self.normal.is_empty()
    }

    fn scheduled(&self) -> u8 {
        self.max.scheduled() + self.hi.scheduled() + self.normal.scheduled()
    }

    fn clear_statistics(&self) {
        self.max.clear_statistics();
        self.hi.clear_statistics();
        self.normal.clear_statistics();
    }

    fn push(&self, task: Self::Task) {
        match task.priority() {
            Priority::Low | Priority::Normal => {
                self.normal.push(task);
            }
            Priority::High => {
                self.hi.push(task);
            }
            Priority::Max => {
                self.max.push(task);
            }
        }
    }

    fn pop(&self) -> Option<Self::Task> {
        const CYCLE_COUNT: u8 = 10;
        const MAX_LIMIT: u8 = 5;
        const HI_LIMIT: u8 = 3;

        loop {
            // If we've reached the end of a cycle, reset the counters
            //
            // A cycle is complete when we've scheduled 10 times.
            //
            // The goal is to preserve the proportion of execution time allocated to each queue
            // relative to the others. So if `max` schedules 3x more than `hi`, and `hi`
            // schedules 2x more than `normal`; then if there are no `max` tasks, we still want
            // `hi` to schedule 2x more than `normal`.
            //
            // To do this, we keep track of the number of schedules from each queue relative to
            // the number of schedules that constitutes a cycle. If the allocated proportion has not
            // been met for a queue with tasks, then that queue gets scheduled first, in order of
            // priority, skipping queues which are empty or which have reached their allocation for
            // a given cycle
            let max_schedules = self.max.scheduled();
            let hi_schedules = self.hi.scheduled();
            let normal_schedules = self.normal.scheduled();
            let schedules = max_schedules + hi_schedules + normal_schedules;
            if schedules > CYCLE_COUNT {
                self.clear_statistics();
            }

            // Max tasks always consume their budget when present
            let has_max = !self.max.is_empty();
            if has_max {
                if max_schedules < MAX_LIMIT {
                    return self.max.pop();
                }
            }

            // Hi tasks always consume their budge when present, plus half of any leftover from `max`
            let has_hi = !self.hi.is_empty();
            let leftover_max_schedules = MAX_LIMIT - max_schedules;
            if has_hi {
                // If we haven't used our time yet, do so now
                if hi_schedules < HI_LIMIT {
                    return self.hi.pop();
                }
                // If there are no `max` tasks, we give half of those ticks to `hi`,
                // giving `hi` tasks a few more cycles
                let extra = (HI_LIMIT + leftover_max_schedules.div_euclid(2)) - hi_schedules;
                if extra > 0 {
                    return self.hi.pop();
                }
            }
            // Normal tasks consume what's left, with low priority tasks only getting 1 schedule every 8 ticks
            // which pull from the normal queue.
            let has_normal = !self.normal.is_empty();
            if has_normal {
                // The normal queue budget is a combination of its minimum requirement + half of the max leftovers
                let extra = (HI_LIMIT
                    + leftover_max_schedules.div_euclid(2)
                    + leftover_max_schedules.rem_euclid(2))
                    - hi_schedules;
                let budget = (2 + extra) - normal_schedules;
                if budget > 0 {
                    let task = self.normal.pop().unwrap();
                    // Check the priority of this task
                    //
                    // If the priority is low, we need to check and see how
                    // many times it has been scheduled since its last execution
                    // time. If it is less than 8 times, we reschedule it for later
                    // and take the next normal task off the queue, if available
                    match task.priority() {
                        Priority::Low => {
                            // If this task is ready to be scheduled, go for it
                            if !task.should_delay() {
                                return Some(task);
                            }
                            // Otherwise, send this back to the global queue so that it
                            // can be picked up again later, and try again
                            self.global.push(task);
                            continue;
                        }
                        _ => return Some(task),
                    }
                }
            }
            // If we reach here, there are either no tasks available,
            // or we are starting back at the highest priority queues, which
            // requires us to reset the counters in order to avoid an infinite loop
            let more = has_max | has_hi | has_normal;
            self.clear_statistics();
            // If we have no more tasks available, try to backfill first
            if !more {
                // If backfilling didn't find anything, there are no more tasks for us
                //
                // We bail out here, but this essentially punts a choice to the scheduler
                // on whether to try and steal or what.
                if !self.backfill() {
                    break;
                }
            }
        }

        None
    }
}

/// A simple task queue for `Process`
pub struct LocalProcessQueue {
    tasks: Worker<Arc<Process>>,
    /// This is safe because the schedules counter is only ever modified
    /// by the same thread, so concurrent reads/writes can never occur
    schedules: UnsafeCell<u8>,
}
impl Default for LocalProcessQueue {
    fn default() -> Self {
        Self {
            tasks: Worker::new_fifo(),
            schedules: UnsafeCell::new(0),
        }
    }
}
impl TaskQueue for LocalProcessQueue {
    type Task = Arc<Process>;

    #[inline]
    fn is_empty(&self) -> bool {
        self.tasks.is_empty()
    }

    #[inline]
    fn push(&self, task: Self::Task) {
        self.tasks.push(task);
    }

    #[inline]
    fn pop(&self) -> Option<Self::Task> {
        unsafe {
            let ptr = self.schedules.get();
            *ptr += 1;
        }
        self.tasks.pop()
    }

    #[inline(always)]
    fn scheduled(&self) -> u8 {
        unsafe { *self.schedules.get() }
    }

    #[inline(always)]
    fn clear_statistics(&self) {
        unsafe {
            let ptr = self.schedules.get();
            *ptr = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::assert_matches::assert_matches;
    use std::collections::btree_map::Entry;
    use std::collections::BTreeMap;
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};

    use firefly_rt::process::Priority;

    struct SimpleTask {
        id: usize,
        schedule_count: AtomicU8,
        priority: Priority,
    }
    impl Default for SimpleTask {
        fn default() -> Self {
            Self::new(Priority::Normal)
        }
    }
    impl SimpleTask {
        fn new(priority: Priority) -> Self {
            static SIMPLE_ID: AtomicUsize = AtomicUsize::new(0);
            let id = SIMPLE_ID.fetch_add(1, Ordering::SeqCst);
            Self {
                id,
                schedule_count: AtomicU8::new(0),
                priority,
            }
        }
    }
    impl Task for SimpleTask {
        type Id = usize;
        fn id(&self) -> Self::Id {
            self.id
        }
        fn priority(&self) -> Priority {
            self.priority
        }
        fn should_delay(&self) -> bool {
            let scheduled = self.schedule_count.fetch_add(1, Ordering::Relaxed);
            if scheduled == 7 {
                self.schedule_count.store(0, Ordering::Relaxed);
                false
            } else {
                true
            }
        }
    }

    #[derive(Default)]
    struct SimpleQueue(UnsafeCell<SimpleQueueInner>);
    impl SimpleQueue {
        #[inline]
        fn inner(&self) -> &SimpleQueueInner {
            unsafe { &*self.0.get() }
        }

        #[inline]
        fn inner_mut(&self) -> &mut SimpleQueueInner {
            unsafe { &mut *self.0.get() }
        }
    }

    #[derive(Default)]
    struct SimpleQueueInner {
        tasks: VecDeque<SimpleTask>,
        schedules: u8,
    }

    impl TaskQueue for SimpleQueue {
        type Task = SimpleTask;

        #[inline]
        fn is_empty(&self) -> bool {
            self.inner().tasks.is_empty()
        }

        #[inline]
        fn push(&self, task: Self::Task) {
            self.inner_mut().tasks.push_back(task);
        }

        #[inline]
        fn pop(&self) -> Option<Self::Task> {
            let inner = self.inner_mut();
            inner.schedules += 1;
            inner.tasks.pop_front()
        }

        #[inline(always)]
        fn scheduled(&self) -> u8 {
            self.inner().schedules
        }

        #[inline(always)]
        fn clear_statistics(&mut self) {
            self.inner_mut().schedules = 0;
        }
    }

    #[derive(Debug, Default)]
    struct ScheduleResult {
        total: usize,
        max: usize,
        hi: usize,
        normal: usize,
        lo: usize,
        seen: BTreeMap<usize, usize>,
    }

    fn run_schedule(max: usize, hi: usize, normal: usize, lo: usize) -> ScheduleResult {
        let mut runq = RunQueue::<SimpleQueue>::default();

        for _ in 0..max {
            runq.push_back(SimpleTask::new(Priority::Max));
        }
        for _ in 0..hi {
            runq.push_back(SimpleTask::new(Priority::High));
        }
        for _ in 0..normal {
            runq.push_back(SimpleTask::default());
        }
        for _ in 0..lo {
            runq.push_back(SimpleTask::new(Priority::Low));
        }

        let mut result = ScheduleResult::default();
        let mut cycle = 0;

        // Run the scheduler 100 cycles, or 1000 ticks
        while let Some(task) = runq.pop() {
            result.total += 1;
            if result.total % 10 == 0 {
                cycle += 1;
            }
            if cycle == 100 {
                break;
            }
            match result.seen.entry(task.id()) {
                Entry::Occupied(mut entry) => {
                    *entry.get_mut() += 1;
                }
                Entry::Vacant(entry) => {
                    entry.insert(1);
                }
            }
            match task.priority() {
                Priority::Low => result.lo += 1,
                Priority::Normal => result.normal += 1,
                Priority::High => result.hi += 1,
                Priority::Max => result.max += 1,
            }
            runq.push_back(task);
        }

        result
    }

    #[test]
    fn run_queue_task_distribution_all_prios_test() {
        // Run a schedule of:
        //
        // * 2 max
        // * 5 hi
        // * 20 normal
        // * 5 low
        let mut result = run_schedule(2, 5, 20, 5);

        assert!((result.max as f64 / result.total as f64) >= 0.5);
        // The proportion of normal priority tasks to other tasks should be ~1:2
        assert!((result.hi as f64 / result.total as f64) >= 0.3);
        // The proportion of normal priority tasks to hi tasks should be ~1:2
        assert!((result.normal as f64 / result.total as f64) >= 0.166);
        // The low priority tasks only get scheduled once every 8 times they are
        // visited in the queue; with only 1000 ticks to this test scheduler, and
        // only 1/6th of them going to normal tasks, and 1/5 of the normal tasks
        // being low priority tasks, each one only 1 shot at being scheduled during
        // the test. We just want to make sure all of them were scheduled
        assert_eq!(result.seen.len(), 32);
        assert_matches!(result.seen.pop_last(), Some((_, 1)));
        assert_matches!(result.seen.pop_last(), Some((_, 1)));
        assert_matches!(result.seen.pop_last(), Some((_, 1)));
        assert_matches!(result.seen.pop_last(), Some((_, 1)));
        assert_matches!(result.seen.pop_last(), Some((_, 1)));
    }

    #[test]
    fn run_queue_task_distribution_lt_max_prios_test() {
        // Run a schedule of:
        //
        // * 5 hi
        // * 20 normal
        // * 5 low
        let mut result = run_schedule(0, 5, 20, 5);

        // The proportion of high priority tasks to other tasks should be ~2:3
        assert!((result.hi as f64 / result.total as f64) >= 0.67);
        // The proportion of normal priority tasks to other tasks should be ~1:3
        assert!((result.normal as f64 / result.total as f64) >= 0.166);
        // The low priority tasks only get scheduled once every 8 times they are
        // visited in the queue; with only 1000 ticks to this test scheduler, and
        // only 1/3rd of them going to normal tasks, and 1/5 of the normal task
        // being low priority tasks, each one gets about 2 shots at being
        // scheduled during the test. We just want to make sure all of them were
        // scheduled at least once
        assert_eq!(result.seen.len(), 30);
        assert_matches!(result.seen.pop_last(), Some((_, n)) if n >= 1);
        assert_matches!(result.seen.pop_last(), Some((_, 2)));
        assert_matches!(result.seen.pop_last(), Some((_, 2)));
        assert_matches!(result.seen.pop_last(), Some((_, 2)));
        assert_matches!(result.seen.pop_last(), Some((_, 2)));
    }

    #[test]
    fn run_queue_task_distribution_lt_hi_prios_test() {
        // Run a schedule of:
        //
        // * 20 normal
        // * 5 low
        let mut result = run_schedule(0, 0, 20, 5);

        // Almost all of the scheduled tasks will be normal tasks, as each
        // low priority task only gets scheduled once out of every 8 cycles,
        // and with 5 tasks out of 25 being low priority, the proportion is
        // even smaller. Each low priority task should have gotten at least
        // 5 schedules though
        assert!((result.normal as f64 / result.total as f64) >= 0.95);
        assert_eq!(result.seen.len(), 25);
        assert_matches!(result.seen.pop_last(), Some((_, n)) if n >= 5);
        assert_matches!(result.seen.pop_last(), Some((_, n)) if n >= 5);
        assert_matches!(result.seen.pop_last(), Some((_, n)) if n >= 5);
        assert_matches!(result.seen.pop_last(), Some((_, n)) if n >= 5);
        assert_matches!(result.seen.pop_last(), Some((_, n)) if n >= 5);
    }

    #[test]
    fn run_queue_task_distribution_lt_normal_prios_test() {
        // Run a schedule of:
        //
        // * 5 low
        let mut result = run_schedule(0, 0, 0, 5);

        // All of the tasks are low priority, and we expect to see them
        // equally serviced here
        assert_eq!(result.lo + 1, result.total);
        assert_eq!(result.seen.len(), 5);
        assert_matches!(result.seen.pop_last(), Some((_, 199)));
        assert_matches!(result.seen.pop_last(), Some((_, 200)));
        assert_matches!(result.seen.pop_last(), Some((_, 200)));
        assert_matches!(result.seen.pop_last(), Some((_, 200)));
        assert_matches!(result.seen.pop_last(), Some((_, 200)));
    }
}
