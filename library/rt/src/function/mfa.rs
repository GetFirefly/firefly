use core::fmt;
use core::str::FromStr;

use static_assertions::assert_eq_size;

use crate::term::Atom;

use super::FunctionSymbol;

/// This struct is a subset of `FunctionSymbol` that is used to more
/// generally represent module/function/arity information for any function
/// whether defined or not.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct ModuleFunctionArity {
    pub module: Atom,
    pub function: Atom,
    pub arity: u8,
}

assert_eq_size!(
    ModuleFunctionArity,
    firefly_bytecode::ModuleFunctionArity<Atom>
);

impl ModuleFunctionArity {
    pub fn new(module: Atom, function: Atom, arity: usize) -> Self {
        Self {
            module,
            function,
            arity: arity.try_into().unwrap(),
        }
    }
}
impl PartialOrd for ModuleFunctionArity {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for ModuleFunctionArity {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.module
            .cmp(&other.module)
            .then_with(|| self.function.cmp(&other.function))
            .then_with(|| self.arity.cmp(&other.arity))
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
impl From<firefly_bytecode::ModuleFunctionArity<Atom>> for ModuleFunctionArity {
    #[inline]
    fn from(mfa: firefly_bytecode::ModuleFunctionArity<Atom>) -> ModuleFunctionArity {
        unsafe { core::mem::transmute(mfa) }
    }
}
impl Into<firefly_bytecode::ModuleFunctionArity<Atom>> for ModuleFunctionArity {
    #[inline]
    fn into(self) -> firefly_bytecode::ModuleFunctionArity<Atom> {
        unsafe { core::mem::transmute(self) }
    }
}
impl FromStr for ModuleFunctionArity {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let Some((module, rest)) = s.split_once(':') else { return Err(()); };
        let Some((function, arity)) = rest.rsplit_once('/') else { return Err(()); };

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
