macro_rules! impl_from {
    ($to:ident :: $constructor:ident ($from:ty)) => {
        impl ::std::convert::From<$from> for $to {
            fn from(x: $from) -> Self {
                $to::$constructor(::std::convert::From::from(x))
            }
        }
    };
}

macro_rules! impl_node {
    ($x:ident <$a:ident, $b:ident>) => {
        impl<$a, $b> crate::code::ast::Node for $x<$a, $b> {
            #[inline]
            fn loc(&self) -> crate::code::ast::Location {
                self.loc
            }
        }
    };
    ($x:ident <$a:ident>) => {
        impl<$a> crate::code::ast::Node for $x<$a> {
            #[inline]
            fn loc(&self) -> crate::code::ast::Location {
                self.loc
            }
        }
    };
    ($x:ty) => {
        impl crate::code::ast::Node for $x {
            #[inline]
            fn loc(&self) -> crate::code::ast::Location {
                self.loc
            }
        }
    };
}

mod clause;
mod expr;
mod form;
mod guard;
mod literal;
mod ty;

pub use self::clause::Clause;
pub use self::expr::*;
pub use self::form::*;
pub use self::guard::OrGuard;
pub use self::literal::*;
pub use self::ty::*;

use firefly_intern::Symbol;

pub type Arity = u8;

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Location {
    pub line: u32,
    pub column: u32,
}
impl From<(u32, u32)> for Location {
    #[inline]
    fn from(loc: (u32, u32)) -> Self {
        Self {
            line: loc.0,
            column: loc.1,
        }
    }
}

pub trait Node {
    fn loc(&self) -> Location;

    #[inline]
    fn line(&self) -> u32 {
        self.loc().line
    }

    #[inline]
    fn column(&self) -> u32 {
        self.loc().column
    }
}

#[derive(Debug, Copy, Clone)]
pub struct FunctionName {
    pub module: Option<Symbol>,
    pub name: Symbol,
    pub arity: Arity,
}
impl FunctionName {
    pub fn new(name: Symbol, arity: Arity) -> Self {
        Self {
            module: None,
            name,
            arity,
        }
    }
}
