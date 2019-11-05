use core::fmt::{self, Display};

use crate::erts::term::Atom;

pub type Arity = u8;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ModuleFunctionArity {
    pub module: Atom,
    pub function: Atom,
    pub arity: Arity,
}

impl Display for ModuleFunctionArity {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "fun {}:{}/{}",
            self.module.name(),
            self.function.name(),
            self.arity
        )
    }
}
