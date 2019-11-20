use alloc::sync::Arc;

use crate::erts::process::code::stack::Trace;
use crate::erts::process::Process;
use crate::erts::term::prelude::Term;

#[derive(Clone, Debug)]
pub enum Stacktrace {
    Term(Term),
    Trace(Trace),
}
impl From<Term> for Stacktrace {
    fn from(term: Term) -> Self {
        Stacktrace::Term(term)
    }
}

impl From<Trace> for Stacktrace {
    fn from(trace: Trace) -> Self {
        Stacktrace::Trace(trace)
    }
}

impl From<&Process> for Stacktrace {
    fn from(process: &Process) -> Self {
        process.stacktrace().into()
    }
}

impl From<&Arc<Process>> for Stacktrace {
    fn from(arc_process: &Arc<Process>) -> Self {
        arc_process.stacktrace().into()
    }
}

impl PartialEq for Stacktrace {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Stacktrace::Term(self_term), Stacktrace::Term(other_term)) => self_term == other_term,
            (Stacktrace::Trace(self_trace), Stacktrace::Trace(other_trace)) => {
                self_trace == other_trace
            }
            _ => unimplemented!("{:?} == {:?}", self, other),
        }
    }
}
