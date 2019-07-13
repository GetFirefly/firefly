use alloc::collections::vec_deque::VecDeque;
use alloc::sync::Arc;

use liblumen_alloc::erts::process::ProcessControlBlock;

use crate::run::Run;

/// A run queue where the `Arc<Process>` is run immediately when it is encountered
#[derive(Default)]
#[cfg_attr(debug_assertions, derive(Debug))]
pub struct Immediate(VecDeque<Arc<ProcessControlBlock>>);

impl Immediate {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn dequeue(&mut self) -> Run {
        match self.0.pop_front() {
            Some(arc_process) => Run::Now(arc_process),
            None => Run::None,
        }
    }

    pub fn enqueue(&mut self, process: Arc<ProcessControlBlock>) {
        self.0.push_back(process);
    }
}
