#![cfg_attr(not(test), allow(dead_code))]

use crate::exception::Exception;
use crate::term::Term;

pub enum Encoding {
    Latin1,
    Unicode,
    Utf8,
}

#[derive(Clone, Copy)]
pub enum Existence {
    DoNotCare,
    Exists,
}

use self::Existence::*;

pub struct Index(pub usize);

pub struct Table {
    names: Vec<String>,
}

impl Table {
    pub fn new() -> Table {
        Table { names: Vec::new() }
    }

    pub fn str_to_index(&mut self, name: &str, existence: Existence) -> Result<Index, Exception> {
        let existing_position = self
            .names
            .iter()
            .position(|existing_name| existing_name == name);

        match (existing_position, existence) {
            (Some(position), _) => Ok(position),
            (None, DoNotCare) => {
                self.names.push(name.to_string());
                Ok(self.names.len() - 1)
            }
            (None, Exists) => {
                let badarg: Term = self.str_to_index("badarg", DoNotCare).unwrap().into();

                Err(error!(badarg))
            }
        }
        .map(|found_or_existing_position| Index(found_or_existing_position))
    }

    pub fn name(&self, index: Index) -> String {
        self.names[index.0].clone()
    }
}
