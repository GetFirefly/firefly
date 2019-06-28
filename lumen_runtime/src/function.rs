use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::code::Code;
use crate::heap::{CloneIntoHeap, Heap};
use crate::process::stack::frame::Frame;
use crate::process::ModuleFunctionArity;
use crate::term::{Tag, Term};

pub struct Function {
    #[allow(dead_code)]
    header: Term,
    module_function_arity: Arc<ModuleFunctionArity>,
    code: Code,
}

impl Function {
    pub fn new(module_function_arity: Arc<ModuleFunctionArity>, code: Code) -> Self {
        Self {
            header: Term {
                tagged: Tag::Function as usize,
            },
            module_function_arity,
            code,
        }
    }

    pub fn frame_with_arguments(&self, argument_vec: Vec<Term>) -> Option<Frame> {
        if argument_vec.len() == self.module_function_arity.arity {
            let mut frame = self.frame();

            for argument in argument_vec.into_iter().rev() {
                frame.push(argument)
            }

            Some(frame)
        } else {
            None
        }
    }

    pub fn module_function_arity(&self) -> Arc<ModuleFunctionArity> {
        Arc::clone(&self.module_function_arity)
    }

    // Private

    fn frame(&self) -> Frame {
        Frame::new(Arc::clone(&self.module_function_arity), self.code)
    }
}

impl CloneIntoHeap for &'static Function {
    fn clone_into_heap(&self, heap: &Heap) -> Self {
        heap.function(self.module_function_arity.clone(), self.code)
    }
}

impl Eq for Function {}

impl Hash for Function {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.module_function_arity.hash(state);
        state.write_usize(self.code as usize);
    }
}

impl Ord for Function {
    fn cmp(&self, other: &Function) -> Ordering {
        (self.module_function_arity.cmp(&other.module_function_arity))
            .then_with(|| (self.code as usize).cmp(&(other.code as usize)))
    }
}

impl PartialEq for Function {
    fn eq(&self, other: &Function) -> bool {
        (self.module_function_arity == other.module_function_arity)
            && ((self.code as usize) == (other.code as usize))
    }
}

impl PartialOrd for Function {
    fn partial_cmp(&self, other: &Function) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
