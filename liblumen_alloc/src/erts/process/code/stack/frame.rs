use core::fmt::{self, Debug, Display};

use crate::erts::process::code::Code;
use crate::erts::term::closure::Definition;
use crate::erts::term::prelude::Atom;
use crate::location::Location;
use crate::Arity;

#[derive(Clone)]
pub struct Frame {
    pub module: Atom,
    pub definition: Definition,
    pub arity: Arity,
    pub location: Location,
    pub code: Code,
}

impl Frame {
    pub fn new(
        module: Atom,
        function: Atom,
        arity: Arity,
        location: Location,
        code: Code,
    ) -> Frame {
        let definition = Definition::Export { function };

        Self::from_definition(module, definition, arity, location, code)
    }

    pub fn from_definition(
        module: Atom,
        definition: Definition,
        arity: Arity,
        location: Location,
        code: Code,
    ) -> Frame {
        Frame {
            module,
            definition,
            arity,
            location,
            code,
        }
    }

    fn code_address(&self) -> usize {
        self.code as usize
    }
}

impl Debug for Frame {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Frame")
            .field("module", &self.module)
            .field("definition", &self.definition)
            .field("arity", &self.arity)
            .field("location", &self.location)
            .field("code", &self.code_address())
            .finish()
    }
}

impl Display for Frame {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{}.{}/{} ({}) at {}",
            self.module,
            self.definition,
            self.arity,
            self.code_address(),
            self.location
        )
    }
}

pub enum Placement {
    Replace,
    Push,
}
