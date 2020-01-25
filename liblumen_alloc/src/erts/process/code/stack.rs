pub mod frame;

use core::fmt::{self, Debug, Display};

use alloc::collections::vec_deque::{Iter, VecDeque};

use self::frame::Frame;

#[derive(Clone, Default)]
pub struct Stack(VecDeque<Frame>);

impl Stack {
    pub fn get(&self, index: usize) -> Option<&Frame> {
        self.0.get(index)
    }

    pub fn iter(&self) -> Iter<Frame> {
        self.0.iter()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn pop(&mut self) -> Option<Frame> {
        self.0.pop_front()
    }

    pub fn push(&mut self, frame: Frame) {
        self.0.push_front(frame);
    }
}

impl Debug for Stack {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for frame in self.iter() {
            writeln!(f, "  {:?}", frame)?;
        }

        Ok(())
    }
}

impl Display for Stack {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for frame in self.iter() {
            writeln!(f, "  {}", frame)?;
        }

        Ok(())
    }
}
