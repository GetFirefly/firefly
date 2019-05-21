use std::collections::vec_deque::VecDeque;
use std::sync::Arc;

use crate::process::Process;
use crate::run::Run;

/// A run queue where the `Arc<Process>` is run immediately when it is encountered
#[derive(Debug, Default)]
pub struct Immediate(VecDeque<Arc<Process>>);

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

    pub fn enqueue(&mut self, process: Arc<Process>) {
        self.0.push_back(process);
    }
}
