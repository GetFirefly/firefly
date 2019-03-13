#![cfg_attr(not(test), allow(dead_code))]

use crate::atom::{self, Existence};
use crate::bad_argument::BadArgument;

pub struct Environment {
    atom_table: atom::Table,
}

impl Environment {
    pub fn new() -> Environment {
        Environment {
            atom_table: atom::Table::new(),
        }
    }

    pub fn atom_index_to_string(&self, atom_index: atom::Index) -> String {
        self.atom_table.name(atom_index)
    }

    pub fn str_to_atom_index(
        &mut self,
        name: &str,
        existence: Existence,
    ) -> Result<atom::Index, BadArgument> {
        self.atom_table.str_to_index(name, existence)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod str_to_atom_index {
        use super::*;

        #[test]
        fn without_same_string_have_different_index() {
            let mut environment = Environment::new();

            assert_ne!(
                environment
                    .str_to_atom_index("true", Existence::DoNotCare)
                    .unwrap()
                    .0,
                environment
                    .str_to_atom_index("false", Existence::DoNotCare)
                    .unwrap()
                    .0
            )
        }

        #[test]
        fn with_same_string_have_same_index() {
            let mut environment = Environment::new();

            assert_eq!(
                environment
                    .str_to_atom_index("atom", Existence::DoNotCare)
                    .unwrap()
                    .0,
                environment
                    .str_to_atom_index("atom", Existence::DoNotCare)
                    .unwrap()
                    .0
            )
        }
    }
}
