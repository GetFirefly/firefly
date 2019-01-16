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
        impl<$a, $b> crate::syntax::ast::ast::Node for $x<$a, $b> {
            fn line(&self) -> crate::syntax::ast::ast::LineNum {
                self.line
            }
        }
    };
    ($x:ident <$a:ident>) => {
        impl<$a> crate::syntax::ast::ast::Node for $x<$a> {
            fn line(&self) -> crate::syntax::ast::ast::LineNum {
                self.line
            }
        }
    };
    ($x:ty) => {
        impl crate::syntax::ast::ast::Node for $x {
            fn line(&self) -> crate::syntax::ast::ast::LineNum {
                self.line
            }
        }
    };
}

pub mod clause;
pub mod common;
pub mod expr;
pub mod form;
pub mod guard;
pub mod literal;
pub mod pat;
pub mod ty;

pub type LineNum = i32;
pub type Arity = u32;

pub trait Node {
    fn line(&self) -> LineNum;
}

#[derive(Debug, Clone)]
pub struct ModuleDecl {
    pub forms: Vec<self::form::Form>,
}
