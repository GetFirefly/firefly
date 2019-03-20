#![cfg_attr(not(test), allow(dead_code))]

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::atom::{self, Existence};
use crate::exception::Exception;
use crate::process::{self, Process};
use crate::term::Term;

pub struct Environment {
    pid_counter: process::identifier::LocalCounter,
    atom_table: atom::Table,
    pub process_by_pid_tagged: HashMap<usize, Arc<RwLock<Process>>>,
}

impl Environment {
    pub fn new() -> Environment {
        Environment {
            atom_table: atom::Table::new(),
            pid_counter: Default::default(),
            process_by_pid_tagged: HashMap::new(),
        }
    }

    pub fn atom_index_to_string(&self, atom_index: atom::Index) -> String {
        self.atom_table.name(atom_index)
    }

    pub fn next_pid(&mut self) -> Term {
        self.pid_counter.next().into()
    }

    pub fn str_to_atom_index(
        &mut self,
        name: &str,
        existence: Existence,
    ) -> Result<atom::Index, Exception> {
        self.atom_table.str_to_index(name, existence)
    }
}

#[cfg(test)]
impl Default for Environment {
    fn default() -> Environment {
        Environment::new()
    }
}

pub fn process(environment_rw_lock: Arc<RwLock<Environment>>) -> Arc<RwLock<Process>> {
    let process = Process::new(Arc::clone(&environment_rw_lock));
    let pid = process.pid;
    let process_rw_lock = Arc::new(RwLock::new(process));

    if let Some(_) = environment_rw_lock
        .write()
        .unwrap()
        .process_by_pid_tagged
        .insert(pid.tagged, Arc::clone(&process_rw_lock))
    {
        panic!("Process already registerd with pid");
    }

    process_rw_lock
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
