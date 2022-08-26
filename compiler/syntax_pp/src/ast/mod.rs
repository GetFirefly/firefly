//! A Rust representation of Abstract Syntax Trees of Erlang modules.
//!
//! Currently, works by loading AST from BEAM files with debug_info enabled
//!
//! # References
//!
//! * [The Abstract Format](http://erlang.org/doc/apps/erts/absform.html)
//!
//! # Examples
//!
//!     use firefly_beam::syntax::ast::AST;
//!
//!     let ast = AST::from_beam_file("tests/testdata/ast/test.beam").unwrap();
//!     println!("{:?}", ast);
mod clause;
mod common;
mod expr;
mod form;
mod guard;
mod literal;
mod pat;
#[cfg(test)]
mod test;
mod ty;

pub use self::clause::*;
pub use self::common::{BinElementTypeSpec, ExternalFun, InternalFun, Nil, Var};
pub use self::form::*;
pub use self::literal::*;
pub use self::pat::*;
pub use self::ty::*;

use std::path::Path;

pub type LineNum = i32;
pub type Arity = u32;

pub trait Node {
    fn line(&self) -> LineNum;
}

/// Abstract Syntax Tree
#[derive(Debug)]
pub struct AST {
    pub module: ModuleDecl,
}
impl AST {
    /// Builds AST from the BEAM file
    pub fn from_beam_file<P: AsRef<Path>>(beam_file: P) -> anyhow::Result<Self> {
        use firefly_beam::beam::AbstractCode;

        let code = AbstractCode::from_beam_file(beam_file)?;
        let forms = crate::parser::from_abstract_code(&code)?;

        Ok(AST {
            module: ModuleDecl { forms },
        })
    }
}
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
