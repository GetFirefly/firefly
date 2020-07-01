use core::fmt::{self, Debug, Display};

use alloc::collections::vec_deque::VecDeque;
use alloc::vec::Vec;

use crate::erts::ModuleFunctionArity;
use crate::process::Frame;

#[derive(Default)]
pub struct Stack(VecDeque<Frame>);

impl Stack {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn pop(&mut self) -> Option<Frame> {
        self.0.pop_front()
    }

    pub fn popn(&mut self, n: usize) {
        for _ in 0..n {
            self.0.pop_front().unwrap();
        }
    }

    pub fn push(&mut self, frame: Frame) {
        self.0.push_front(frame);
    }

    pub fn top(&self) -> Option<&Frame> {
        self.0.get(0)
    }

    pub fn trace(&self) -> Trace {
        let mut stacktrace = Vec::with_capacity(self.len());

        for frame in self.0.iter() {
            stacktrace.push(frame.module_function_arity())
        }

        Trace(stacktrace)
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

pub struct Trace(Vec<ModuleFunctionArity>);

impl Debug for Trace {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for module_function_arity in self.0.iter() {
            writeln!(f, "  {}", module_function_arity)?;
        }

        Ok(())
    }
}

impl Display for Trace {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for module_function_arity in self.0.iter() {
            writeln!(f, "  {}", module_function_arity)?;
        }

        Ok(())
    }
}
