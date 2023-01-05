use core::fmt;
use core::str::FromStr;

use crate::term::Atom;

use super::FunctionSymbol;

pub type Arity = u8;

/// This struct is a subset of `FunctionSymbol` that is used to more
/// generally represent module/function/arity information for any function
/// whether defined or not.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ModuleFunctionArity {
    pub module: Atom,
    pub function: Atom,
    pub arity: Arity,
}
impl ModuleFunctionArity {
    pub fn new(module: Atom, function: Atom, arity: usize) -> Self {
        Self {
            module,
            function,
            arity: arity.try_into().unwrap(),
        }
    }
}
impl From<FunctionSymbol> for ModuleFunctionArity {
    #[inline]
    fn from(sym: FunctionSymbol) -> Self {
        Self {
            module: sym.module,
            function: sym.function,
            arity: sym.arity,
        }
    }
}
impl FromStr for ModuleFunctionArity {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let Some((module, rest)) = s.split_once(':') else { return Err(()); };
        let Some((function, arity)) = rest.split_once('/') else { return Err(()); };

        let module = Atom::try_from(module).unwrap();
        let function = Atom::try_from(function).unwrap();
        let Ok(arity) = arity.parse::<u8>() else { return Err(()); };

        Ok(Self {
            module,
            function,
            arity,
        })
    }
}
impl fmt::Debug for ModuleFunctionArity {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self)
    }
}
impl fmt::Display for ModuleFunctionArity {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}:{}/{}", self.module, self.function, self.arity)
    }
}
