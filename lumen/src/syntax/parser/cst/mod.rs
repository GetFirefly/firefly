pub mod clauses;
pub mod commons;
pub mod exprs;
pub mod forms;
pub mod guard_tests;
pub mod patterns;
pub mod types;

mod expr;
mod form;
mod guard_test;
mod literal;
mod pattern;
mod ty;

use trackable::track;

use crate::syntax::parser::traits::{Parse, TokenRead};
use crate::syntax::parser::{Parser, Result};

pub use self::expr::Expr;
pub use self::form::Form;
pub use self::guard_test::GuardTest;
pub use self::literal::Literal;
pub use self::pattern::Pattern;
pub use self::ty::Type;

/// `Vec<Form>`
#[derive(Debug, Clone)]
pub struct ModuleDecl {
    pub forms: Vec<Form>,
}
impl Parse for ModuleDecl {
    fn parse<T>(parser: &mut Parser<T>) -> Result<Self>
    where
        T: TokenRead,
    {
        let mut forms = Vec::new();
        while !track!(parser.eos())? {
            let form = track!(parser.parse())?;
            forms.push(form);
        }
        Ok(ModuleDecl { forms })
    }
}
