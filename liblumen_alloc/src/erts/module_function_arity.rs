use core::convert::AsRef;
use core::fmt::{self, Display};
use core::str::FromStr;

use crate::erts::term::prelude::Atom;

pub type Arity = u8;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ModuleFunctionArity {
    pub module: Atom,
    pub function: Atom,
    pub arity: Arity,
}
impl ModuleFunctionArity {
    #[inline]
    pub fn from_symbol_name(s: impl AsRef<str>) -> Result<Self, ()> {
        s.as_ref().parse::<Self>()
    }
}

impl FromStr for ModuleFunctionArity {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut parts = s.splitn(2, ':');
        let module = parts.next();
        let function_and_arity = parts.next();
        let (function, arity) = function_and_arity
            .map(|fa| {
                let mut parts = fa.splitn(2, '/');
                (parts.next(), parts.next())
            })
            .unwrap_or_else(|| (None, None));

        match (module, function, arity) {
            (Some(m), Some(f), Some(a)) => {
                let arity = u8::from_str(a).map_err(|_| ())?;
                Ok(ModuleFunctionArity {
                    module: Atom::from_str(m),
                    function: Atom::from_str(f),
                    arity,
                })
            }
            _ => Err(()),
        }
    }
}

impl Display for ModuleFunctionArity {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{}:{}/{}",
            self.module.name(),
            self.function.name(),
            self.arity
        )
    }
}
