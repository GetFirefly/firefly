#![cfg_attr(not(test), allow(dead_code))]
///! The memory specific to a process in the VM.
use std::sync::Arc;
use std::sync::RwLock;

use crate::environment::Environment;
use crate::list::Cons;
use crate::term::{Tag, Term};

pub struct Process {
    environment: Arc<RwLock<Environment>>,
}

impl Process {
    pub fn new(environment: Arc<RwLock<Environment>>) -> Process {
        Process { environment }
    }

    /// Combines the two `Term`s into a list `Term`.  The list is only a proper list if the `tail`
    /// is a list `Term` (`Term.tag` is `Tag::List`) or empty list (`Term.tag` is `Tag::EmptyList`).
    pub fn cons(&self, head: Term, tail: Term) -> *const Cons {
        Box::leak(Box::new(Cons::new(head, tail)))
    }

    pub fn atom_to_string(&self, term: &Term) -> String {
        assert_eq!(term.tag(), Tag::Atom);

        self.environment.read().unwrap().atom_to_string(term)
    }

    pub fn find_or_insert_atom(&mut self, name: &str) -> Term {
        self.environment.write().unwrap().find_or_insert_atom(name)
    }
}

/// Like `std::convert::TryInto`, but additionally takes `&mut Process` in case it is needed to
/// lookup or create new values in the `Process`.
pub trait TryIntoProcess<T> {
    /// The type return in the event of a conversion error.
    type Error;

    /// Performs the conversion.
    fn try_into_process(self, process: &mut Process) -> Result<T, Self::Error>;
}

/// Like `std::convert::Into`, but additionally takes `&mut Process` in case it is needed to
/// lookup or create new values in the `Process`.
pub trait IntoProcess<T> {
    /// Performs the conversion.
    fn into_process(self, process: &mut Process) -> T;
}

impl IntoProcess<Term> for bool {
    fn into_process(self: Self, process: &mut Process) -> Term {
        process.find_or_insert_atom(&self.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod find_or_insert_atom {
        use super::*;

        #[test]
        fn have_atom_tags() {
            let mut process = process();

            assert_eq!(process.find_or_insert_atom("true").tag(), Tag::Atom);
            assert_eq!(process.find_or_insert_atom("false").tag(), Tag::Atom);
        }

        #[test]
        fn with_same_string_have_same_tagged_value() {
            let mut process = process();

            assert_eq!(
                process.find_or_insert_atom("atom").tagged,
                process.find_or_insert_atom("atom").tagged
            )
        }
    }

    fn process() -> Process {
        Process::new(Arc::new(RwLock::new(Environment::new())))
    }
}
