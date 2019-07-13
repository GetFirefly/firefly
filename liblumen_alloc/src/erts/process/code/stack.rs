pub mod frame;

#[cfg(debug_assertions)]
use core::fmt::{self, Debug};

use alloc::collections::vec_deque::{Iter, VecDeque};

use self::frame::Frame;

#[derive(Default)]
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

#[cfg(debug_assertions)]
impl Debug for Stack {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "{{")?;
        writeln!(f, "  let stack: Stack = Default::default();")?;

        for frame in self.0.iter().rev() {
            writeln!(f, "  stack.push({:?});", frame)?;
        }

        writeln!(f, "  stack")?;
        write!(f, "}}")
    }
}
