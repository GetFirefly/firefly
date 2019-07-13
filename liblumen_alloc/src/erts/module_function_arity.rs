use core::fmt::{self, Display};

use crate::erts::term::Atom;

#[derive(Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[cfg_attr(debug_assertions, derive(Debug))]
pub struct ModuleFunctionArity {
    pub module: Atom,
    pub function: Atom,
    pub arity: u8,
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
