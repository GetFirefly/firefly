mod attributes;
mod expr;
mod functions;
mod module;
mod types;

use firefly_diagnostics::{SourceIndex, Spanned};
use firefly_intern::{Ident, Symbol};

pub use self::attributes::*;
pub use self::expr::*;
pub use self::functions::*;
pub use self::module::*;
pub use self::types::*;

use crate::lexer::Token;
use crate::preprocessor::PreprocessorError;

/// Used for AST functions which need to raise an error to the parser directly
pub type TryParseResult<T> =
    Result<T, lalrpop_util::ParseError<SourceIndex, Token, PreprocessorError>>;

/// Represents either a concrete name (an atom) or a variable name (an identifier).
/// This is used in constructs where either are permitted.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Spanned)]
pub enum Name {
    Atom(Ident),
    Var(Ident),
}
impl Name {
    pub fn symbol(&self) -> Symbol {
        match self {
            Name::Atom(Ident { ref name, .. }) => name.clone(),
            Name::Var(Ident { ref name, .. }) => name.clone(),
        }
    }

    pub fn atom(&self) -> Ident {
        match self {
            Name::Atom(ident) => *ident,
            _ => panic!(),
        }
    }

    pub fn is_atom(&self) -> bool {
        match self {
            Name::Atom(_) => true,
            _ => false,
        }
    }

    pub fn var(&self) -> Ident {
        match self {
            Name::Var(ident) => *ident,
            _ => panic!(),
        }
    }

    pub fn is_var(&self) -> bool {
        match self {
            Name::Var(_) => true,
            _ => false,
        }
    }

    pub fn ident(&self) -> Ident {
        match self {
            Name::Atom(ident) => *ident,
            Name::Var(ident) => *ident,
        }
    }
}
impl PartialOrd for Name {
    fn partial_cmp(&self, other: &Name) -> Option<std::cmp::Ordering> {
        self.symbol().partial_cmp(&other.symbol())
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd)]
pub enum Arity {
    Int(u8),
    Var(Ident),
}
