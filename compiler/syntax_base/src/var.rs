use std::cmp::Ordering;
use std::fmt;

use liblumen_diagnostics::Spanned;
use liblumen_intern::{symbols, Ident, Symbol};

use crate::*;

#[derive(Clone, Spanned)]
pub struct Var {
    pub annotations: Annotations,
    #[span]
    pub name: Ident,
    /// Used to represent function variables
    pub arity: Option<usize>,
}
annotated!(Var);
impl fmt::Debug for Var {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use std::fmt::Write;

        match self.arity {
            None => write!(f, "Var({}", self.name.name)?,
            Some(arity) => write!(f, "Var({}/{}", self.name.name, arity)?,
        }
        if self.annotations.is_empty() {
            f.write_char(')')
        } else {
            write!(f, " {:?})", &self.annotations)
        }
    }
}
impl Var {
    pub fn new(name: Ident) -> Self {
        Self {
            annotations: Annotations::default(),
            name,
            arity: None,
        }
    }

    pub fn new_with_arity(name: Ident, arity: usize) -> Self {
        Self {
            annotations: Annotations::default(),
            name,
            arity: Some(arity),
        }
    }

    pub fn name(&self) -> Symbol {
        self.name.name
    }

    #[inline]
    pub fn is_wildcard(&self) -> bool {
        self.name == symbols::Underscore
    }
}
impl Eq for Var {}
impl PartialEq for Var {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.arity == other.arity
    }
}
impl PartialOrd for Var {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Var {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        match self.name.cmp(&other.name) {
            Ordering::Equal if self.arity.is_none() && other.arity.is_none() => Ordering::Equal,
            Ordering::Equal if self.arity.is_some() && other.arity.is_some() => {
                self.arity.unwrap().cmp(&other.arity.unwrap())
            }
            other => other,
        }
    }
}
