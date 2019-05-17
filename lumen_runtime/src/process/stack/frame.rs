use std::collections::vec_deque::VecDeque;
use std::fmt::{self, Debug};
use std::sync::Arc;

use crate::code::Code;
use crate::process::ModuleFunctionArity;
use crate::term::Term;

pub struct Frame {
    module_function_arity: Arc<ModuleFunctionArity>,
    code: Code,
    data: VecDeque<Term>,
}

impl Frame {
    pub fn new(module_function_arity: Arc<ModuleFunctionArity>, code: Code) -> Frame {
        Frame {
            module_function_arity,
            code,
            data: Default::default(),
        }
    }

    pub fn module_function_arity(&self) -> Arc<ModuleFunctionArity> {
        Arc::clone(&self.module_function_arity)
    }

    pub fn code(&self) -> Code {
        self.code
    }

    pub fn get(&self, index: usize) -> Option<&Term> {
        self.data.get(index)
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut Term> {
        self.data.get_mut(index)
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn pop(&mut self) -> Option<Term> {
        self.data.pop_front()
    }

    pub fn push(&mut self, term: Term) {
        self.data.push_front(term);
    }
}

impl Debug for Frame {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "{{")?;
        writeln!(
            f,
            "  let frame = Frame::new({:?}, ...);",
            self.module_function_arity
        )?;

        for datum in self.data.iter().rev() {
            writeln!(f, "  frame.push({:?});", datum)?;
        }

        writeln!(f, "  frame")?;
        write!(f, "}}")
    }
}
