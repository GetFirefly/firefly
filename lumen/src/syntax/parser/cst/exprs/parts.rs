use trackable::track;

use crate::syntax::tokenizer::tokens::{KeywordToken, SymbolToken};
use crate::syntax::tokenizer::values::{Keyword, Symbol};
use crate::syntax::tokenizer::{Position, PositionRange};

use crate::syntax::parser::traits::{Parse, TokenRead};
use crate::syntax::parser::{Parser, Result};

use super::super::clauses::{CaseClause, CatchClause};
use super::super::commons::parts::{Clauses, Sequence};
use super::super::Expr;
use super::super::Pattern;

/// `Sequence<Expr>`
#[derive(Debug, Clone)]
pub struct Body {
    pub exprs: Sequence<Expr>,
}
impl Parse for Body {
    fn parse<T: TokenRead>(parser: &mut Parser<T>) -> Result<Self> {
        let exprs = track!(parser.parse())?;
        Ok(Body { exprs })
    }
}
impl PositionRange for Body {
    fn start_position(&self) -> Position {
        self.exprs.start_position()
    }
    fn end_position(&self) -> Position {
        self.exprs.end_position()
    }
}

/// `Generator` | `Filter`
#[derive(Debug, Clone)]
pub enum Qualifier {
    Generator(Generator),
    Filter(Expr),
}
impl Parse for Qualifier {
    fn parse<T: TokenRead>(parser: &mut Parser<T>) -> Result<Self> {
        if let Ok(generator) = parser.transaction(|parser| parser.parse()) {
            Ok(Qualifier::Generator(generator))
        } else {
            Ok(Qualifier::Filter(track!(parser.parse())?))
        }
    }
}
impl PositionRange for Qualifier {
    fn start_position(&self) -> Position {
        match *self {
            Qualifier::Generator(ref x) => x.start_position(),
            Qualifier::Filter(ref x) => x.start_position(),
        }
    }
    fn end_position(&self) -> Position {
        match *self {
            Qualifier::Generator(ref x) => x.end_position(),
            Qualifier::Filter(ref x) => x.end_position(),
        }
    }
}

/// `Pattern` (`<-`|`<=`) `Expr`
#[derive(Debug, Clone)]
pub struct Generator {
    pub pattern: Pattern,
    pub _arrow: SymbolToken,
    pub source: Expr,
}
impl Parse for Generator {
    fn parse<T: TokenRead>(parser: &mut Parser<T>) -> Result<Self> {
        Ok(Generator {
            pattern: track!(parser.parse())?,
            _arrow: track!(parser.expect_any(&[&Symbol::LeftArrow, &Symbol::DoubleLeftArrow],))?,
            source: track!(parser.parse())?,
        })
    }
}
impl PositionRange for Generator {
    fn start_position(&self) -> Position {
        self.pattern.start_position()
    }
    fn end_position(&self) -> Position {
        self.source.end_position()
    }
}

/// `after` `Expr` `->` `Body`
#[derive(Debug, Clone)]
pub struct Timeout {
    pub _after: KeywordToken,
    pub duration: Expr,
    pub _arrow: SymbolToken,
    pub body: Body,
}
impl Parse for Timeout {
    fn parse<T: TokenRead>(parser: &mut Parser<T>) -> Result<Self> {
        Ok(Timeout {
            _after: track!(parser.expect(&Keyword::After))?,
            duration: track!(parser.parse())?,
            _arrow: track!(parser.expect(&Symbol::RightArrow))?,
            body: track!(parser.parse())?,
        })
    }
}
impl PositionRange for Timeout {
    fn start_position(&self) -> Position {
        self._after.start_position()
    }
    fn end_position(&self) -> Position {
        self.body.end_position()
    }
}

/// `of` `Clauses<CaseClause>`
#[derive(Debug, Clone)]
pub struct TryOf {
    pub _of: KeywordToken,
    pub clauses: Clauses<CaseClause>,
}
impl Parse for TryOf {
    fn parse<T: TokenRead>(parser: &mut Parser<T>) -> Result<Self> {
        Ok(TryOf {
            _of: track!(parser.expect(&Keyword::Of))?,
            clauses: track!(parser.parse())?,
        })
    }
}
impl PositionRange for TryOf {
    fn start_position(&self) -> Position {
        self._of.start_position()
    }
    fn end_position(&self) -> Position {
        self.clauses.end_position()
    }
}

/// `catch` `Clauses<CatchClause>`a
#[derive(Debug, Clone)]
pub struct TryCatch {
    pub _catch: KeywordToken,
    pub clauses: Clauses<CatchClause>,
}
impl Parse for TryCatch {
    fn parse<T: TokenRead>(parser: &mut Parser<T>) -> Result<Self> {
        Ok(TryCatch {
            _catch: track!(parser.expect(&Keyword::Catch))?,
            clauses: track!(parser.parse())?,
        })
    }
}
impl PositionRange for TryCatch {
    fn start_position(&self) -> Position {
        self._catch.start_position()
    }
    fn end_position(&self) -> Position {
        self.clauses.end_position()
    }
}

/// `after` `Body`
#[derive(Debug, Clone)]
pub struct TryAfter {
    pub _after: KeywordToken,
    pub body: Body,
}
impl Parse for TryAfter {
    fn parse<T: TokenRead>(parser: &mut Parser<T>) -> Result<Self> {
        Ok(TryAfter {
            _after: track!(parser.expect(&Keyword::After))?,
            body: track!(parser.parse())?,
        })
    }
}
impl PositionRange for TryAfter {
    fn start_position(&self) -> Position {
        self._after.start_position()
    }
    fn end_position(&self) -> Position {
        self.body.end_position()
    }
}
