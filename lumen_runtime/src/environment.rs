#![cfg_attr(not(test), allow(dead_code))]

use std::convert::TryInto;

use crate::atom;
use crate::term::{AtomIndexOverflow, Tag, Term};

pub struct Environment {
    atom_table: atom::Table,
}

impl Environment {
    pub fn new() -> Environment {
        Environment {
            atom_table: atom::Table::new(),
        }
    }

    pub fn atom_to_string(&self, term: &Term) -> String {
        assert_eq!(term.tag(), Tag::Atom);

        self.atom_table.name(term.into())
    }

    pub fn find_or_insert_atom(&mut self, name: &str) -> Result<Term, AtomIndexOverflow> {
        self.atom_table.find_or_insert(name).try_into()
    }
}

/// Like `std::convert::TryInto`, but additionally takes `&mut Environment` in case it is needed to
/// lookup or create new values in the `Environment`.
pub trait TryIntoEnvironment<T> {
    /// THe type reutrn in the event of a conversion error.
    type Error;

    /// Performs the conversion.
    fn try_into_environment(self, environment: &mut Environment) -> Result<T, Self::Error>;
}

#[cfg(test)]
mod tests {
    use super::*;

    mod find_or_insert_atom {
        use super::*;

        #[test]
        fn have_atom_tags() {
            let mut environment = Environment::new();
            assert_eq!(
                environment.find_or_insert_atom("true").unwrap().tag(),
                Tag::Atom
            );
            assert_eq!(
                environment.find_or_insert_atom("false").unwrap().tag(),
                Tag::Atom
            );
        }

        #[test]
        fn with_same_string_have_same_tagged_value() {
            let mut environment = Environment::new();
            assert_eq!(
                environment.find_or_insert_atom("atom").unwrap().tagged,
                environment.find_or_insert_atom("atom").unwrap().tagged
            )
        }
    }
}
