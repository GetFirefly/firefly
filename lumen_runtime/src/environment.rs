#![cfg_attr(not(test), allow(dead_code))]

use crate::atom;
use crate::term::{Tag, Term};

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

    pub fn str_to_atom(&mut self, name: &str) -> Term {
        self.atom_table.str_to_index(name).into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod find_or_insert_atom {
        use super::*;

        #[test]
        fn have_atom_tags() {
            let mut environment = Environment::new();
            assert_eq!(environment.str_to_atom("true").tag(), Tag::Atom);
            assert_eq!(environment.str_to_atom("false").tag(), Tag::Atom);
        }

        #[test]
        fn with_same_string_have_same_tagged_value() {
            let mut environment = Environment::new();
            assert_eq!(
                environment.str_to_atom("atom").tagged,
                environment.str_to_atom("atom").tagged
            )
        }
    }
}
