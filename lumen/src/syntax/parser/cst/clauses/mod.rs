pub mod parts;

use trackable::track;

use crate::syntax::tokenizer::tokens::{AtomToken, SymbolToken, VariableToken};
use crate::syntax::tokenizer::values::Symbol;
use crate::syntax::tokenizer::{Position, PositionRange};

use crate::syntax::parser::traits::{Parse, TokenRead};
use crate::syntax::parser::{Parser, Result};

use super::commons::parts::{Args, Clauses, Sequence};
use super::exprs::parts::Body;
use super::types;
use super::{GuardTest, Pattern, Type};

use self::parts::{ExceptionClass, WhenGuard};

/// `Option<ExceptionClass>` `Pattern` `Option<WhenGuard>` `->` `Body`
#[derive(Debug, Clone)]
pub struct CatchClause {
    pub class: Option<ExceptionClass>,
    pub pattern: Pattern,
    pub guard: Option<WhenGuard>,
    pub _arrow: SymbolToken,
    pub body: Body,
}
impl Parse for CatchClause {
    fn parse<T>(parser: &mut Parser<T>) -> Result<Self>
    where
        T: TokenRead,
    {
        Ok(CatchClause {
            class: track!(parser.parse())?,
            pattern: track!(parser.parse())?,
            guard: track!(parser.parse())?,
            _arrow: track!(parser.expect(&Symbol::RightArrow))?,
            body: track!(parser.parse())?,
        })
    }
}
impl PositionRange for CatchClause {
    fn start_position(&self) -> Position {
        self.class
            .as_ref()
            .map(|x| x.start_position())
            .unwrap_or_else(|| self.pattern.start_position())
    }
    fn end_position(&self) -> Position {
        self.body.end_position()
    }
}

/// `Args<Type>` `->` `Type` `Option<Constraints>`
#[derive(Debug, Clone)]
pub struct SpecClause {
    pub args: Args<Type>,
    pub _arrow: SymbolToken,
    pub return_type: Type,
    pub constraints: Option<types::Constraints>,
}
impl Parse for SpecClause {
    fn parse<T>(parser: &mut Parser<T>) -> Result<Self>
    where
        T: TokenRead,
    {
        Ok(SpecClause {
            args: track!(parser.parse())?,
            _arrow: track!(parser.expect(&Symbol::RightArrow))?,
            return_type: track!(parser.parse())?,
            constraints: track!(parser.parse())?,
        })
    }
}
impl PositionRange for SpecClause {
    fn start_position(&self) -> Position {
        self.args.start_position()
    }
    fn end_position(&self) -> Position {
        self.constraints
            .as_ref()
            .map(|t| t.end_position())
            .unwrap_or_else(|| self.return_type.end_position())
    }
}

/// `Pattern` `Option<WhenGuard>` `->` `Body`
#[derive(Debug, Clone)]
pub struct CaseClause {
    pub pattern: Pattern,
    pub guard: Option<WhenGuard>,
    pub _arrow: SymbolToken,
    pub body: Body,
}
impl Parse for CaseClause {
    fn parse<T>(parser: &mut Parser<T>) -> Result<Self>
    where
        T: TokenRead,
    {
        Ok(CaseClause {
            pattern: track!(parser.parse())?,
            guard: track!(parser.parse())?,
            _arrow: track!(parser.expect(&Symbol::RightArrow))?,
            body: track!(parser.parse())?,
        })
    }
}
impl PositionRange for CaseClause {
    fn start_position(&self) -> Position {
        self.pattern.start_position()
    }
    fn end_position(&self) -> Position {
        self.body.end_position()
    }
}

/// `Clauses<Sequence<GuardTest>>` `->` `Body`
#[derive(Debug, Clone)]
pub struct IfClause {
    pub guard: Clauses<Sequence<GuardTest>>,
    pub _arrow: SymbolToken,
    pub body: Body,
}
impl Parse for IfClause {
    fn parse<T>(parser: &mut Parser<T>) -> Result<Self>
    where
        T: TokenRead,
    {
        Ok(IfClause {
            guard: track!(parser.parse())?,
            _arrow: track!(parser.expect(&Symbol::RightArrow))?,
            body: track!(parser.parse())?,
        })
    }
}
impl PositionRange for IfClause {
    fn start_position(&self) -> Position {
        self.guard.start_position()
    }
    fn end_position(&self) -> Position {
        self.body.end_position()
    }
}

/// `Args<Pattern>` `Option<WhenGuard>` `->` `Body`
#[derive(Debug, Clone)]
pub struct FunClause {
    pub patterns: Args<Pattern>,
    pub guard: Option<WhenGuard>,
    pub _arrow: SymbolToken,
    pub body: Body,
}
impl Parse for FunClause {
    fn parse<T>(parser: &mut Parser<T>) -> Result<Self>
    where
        T: TokenRead,
    {
        Ok(FunClause {
            patterns: track!(parser.parse())?,
            guard: track!(parser.parse())?,
            _arrow: track!(parser.expect(&Symbol::RightArrow))?,
            body: track!(parser.parse())?,
        })
    }
}
impl PositionRange for FunClause {
    fn start_position(&self) -> Position {
        self.patterns.start_position()
    }
    fn end_position(&self) -> Position {
        self.body.end_position()
    }
}

/// `VariableToken` `Args<Pattern>` `Option<WhenGuard>` `->` `Body`
#[derive(Debug, Clone)]
pub struct NamedFunClause {
    pub name: VariableToken,
    pub patterns: Args<Pattern>,
    pub guard: Option<WhenGuard>,
    pub _arrow: SymbolToken,
    pub body: Body,
}
impl Parse for NamedFunClause {
    fn parse<T>(parser: &mut Parser<T>) -> Result<Self>
    where
        T: TokenRead,
    {
        Ok(NamedFunClause {
            name: track!(parser.parse())?,
            patterns: track!(parser.parse())?,
            guard: track!(parser.parse())?,
            _arrow: track!(parser.expect(&Symbol::RightArrow))?,
            body: track!(parser.parse())?,
        })
    }
}
impl PositionRange for NamedFunClause {
    fn start_position(&self) -> Position {
        self.name.start_position()
    }
    fn end_position(&self) -> Position {
        self.body.end_position()
    }
}

/// `AtomToken` `Args<Pattern>` `Option<WhenGuard>` `->` `Body`
#[derive(Debug, Clone)]
pub struct FunDeclClause {
    pub name: AtomToken,
    pub patterns: Args<Pattern>,
    pub guard: Option<WhenGuard>,
    pub _arrow: SymbolToken,
    pub body: Body,
}
impl Parse for FunDeclClause {
    fn parse<T>(parser: &mut Parser<T>) -> Result<Self>
    where
        T: TokenRead,
    {
        // TODO: handle predefined macros
        Ok(FunDeclClause {
            name: track!(parser.parse())?,
            patterns: track!(parser.parse())?,
            guard: track!(parser.parse())?,
            _arrow: track!(parser.expect(&Symbol::RightArrow))?,
            body: track!(parser.parse())?,
        })
    }
}
impl PositionRange for FunDeclClause {
    fn start_position(&self) -> Position {
        self.name.start_position()
    }
    fn end_position(&self) -> Position {
        self.body.end_position()
    }
}
