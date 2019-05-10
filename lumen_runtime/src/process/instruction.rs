use std::sync::Arc;

use crate::code::apply;
use crate::process::Process;
use crate::term::Term;

pub enum Instruction {
    Apply {
        module: Term,
        function: Term,
        arguments: Vec<Term>,
    },
}

use Instruction::*;

impl Instruction {
    pub fn run(self, arc_process: &Arc<Process>) -> bool {
        match self {
            Apply {
                module,
                function,
                arguments,
            } => {
                match apply(module, function, arguments, arc_process) {
                    Ok(term) => {
                        arc_process.stack.lock().unwrap().push(term);

                        true
                    }
                    Err(exception) => {
                        // TODO try/catch
                        arc_process.exception(exception);

                        false
                    }
                }
            }
        }
    }
}
