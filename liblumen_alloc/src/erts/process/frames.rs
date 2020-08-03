mod queue;
mod stack;

use super::frame::Frame;
use super::frame_with_arguments::FrameWithArguments;

pub use self::queue::Drain;
use self::queue::Queue;
use self::stack::Stack;
pub use self::stack::Trace as StackTrace;

#[derive(Default)]
pub struct Frames {
    /// The stack of frames that are currently executing in the process
    stack: Stack,
    /// The future frames that will be pushed on onto the `stack` when the current top `Frame`
    /// returns.
    queue: Queue,
}

impl Frames {
    pub fn current(&self) -> Option<&Frame> {
        self.stack.top()
    }

    pub fn push(&mut self, frame: Frame) {
        self.stack.push(frame);
    }

    pub fn pop(&mut self) -> Option<Frame> {
        self.stack.pop()
    }

    pub fn popn(&mut self, n: usize) {
        self.stack.popn(n)
    }

    /// Queue a future `frame` to run once `Frame` at top of `stack` returns.
    pub fn queue(&mut self, frame_with_arguments: FrameWithArguments) {
        self.queue.push(frame_with_arguments);
    }

    pub fn drain_queue(&mut self) -> Vec<FrameWithArguments> {
        self.queue.drain().collect()
    }

    pub fn stacktrace(&self) -> StackTrace {
        self.stack.trace()
    }
}
