use core::convert::{TryFrom, TryInto};

use crate::erts::exception::runtime;
use crate::erts::term::{Atom, Term, TypedTerm};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Priority {
    Low,
    Normal,
    High,
    Max,
}

impl Default for Priority {
    fn default() -> Priority {
        Priority::Normal
    }
}

impl TryFrom<Atom> for Priority {
    type Error = runtime::Exception;

    fn try_from(atom: Atom) -> Result<Self, Self::Error> {
        match atom.name() {
            "low" => Ok(Priority::Low),
            "normal" => Ok(Priority::Normal),
            "high" => Ok(Priority::High),
            "max" => Ok(Priority::Max),
            _ => Err(badarg!()),
        }
    }
}

impl TryFrom<Term> for Priority {
    type Error = runtime::Exception;

    fn try_from(term: Term) -> Result<Self, Self::Error> {
        term.to_typed_term().unwrap().try_into()
    }
}

impl TryFrom<TypedTerm> for Priority {
    type Error = runtime::Exception;

    fn try_from(typed_term: TypedTerm) -> Result<Self, Self::Error> {
        match typed_term {
            TypedTerm::Atom(atom) => atom.try_into(),
            _ => Err(badarg!()),
        }
    }
}
