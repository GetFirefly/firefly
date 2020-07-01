pub use alloc::collections::vec_deque::Drain;
use alloc::collections::vec_deque::VecDeque;

use crate::erts::process::FrameWithArguments;

#[derive(Default)]
pub struct Queue(VecDeque<FrameWithArguments>);

impl Queue {
    pub fn drain(&mut self) -> Drain<FrameWithArguments> {
        self.0.drain(..)
    }

    pub fn push(&mut self, frame_with_arguments: FrameWithArguments) {
        self.0.push_back(frame_with_arguments);
    }
}
