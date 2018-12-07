pub mod parts;

use trackable::track;

use crate::syntax::tokenizer::tokens::{KeywordToken, SymbolToken};
use crate::syntax::tokenizer::values::{Keyword, Symbol};
use crate::syntax::tokenizer::{Position, PositionRange};

use crate::syntax::parser::traits::{Parse, ParseTail, TokenRead};
use crate::syntax::parser::{Parser, Result};

use super::clauses::{CaseClause, FunClause, IfClause, NamedFunClause};
use super::commons::parts::{Clauses, ModulePrefix, NameAndArity, Sequence};
use super::commons::{self, AtomOrVariable, IntegerOrVariable};
use super::Expr;

use self::parts::{Body, Qualifier, Timeout, TryAfter, TryCatch, TryOf};

pub type Tuple = commons::Tuple<Expr>;
pub type Map = commons::Map<Expr>;
pub type Record = commons::Record<Expr>;
pub type RecordFieldIndex = commons::RecordFieldIndex;
pub type List = commons::List<Expr>;
pub type Bits = commons::Bits<Expr>;
pub type Parenthesized = commons::Parenthesized<Expr>;
pub type FunCall = commons::Call<Expr>;
pub type UnaryOpCall = commons::UnaryOpCall<Expr>;
pub type BinaryOpCall = commons::BinaryOpCall<Expr>;
pub type Match = commons::Match<Expr>;

/// `Expr` `Map`
#[derive(Debug, Clone)]
pub struct MapUpdate {
    pub map: Expr,
    pub update: Map,
}
impl ParseTail for MapUpdate {
    type Head = Expr;
    fn parse_tail<T: TokenRead>(parser: &mut Parser<T>, head: Self::Head) -> Result<Self> {
        Ok(MapUpdate {
            map: head,
            update: track!(parser.parse())?,
        })
    }
}
impl PositionRange for MapUpdate {
    fn start_position(&self) -> Position {
        self.map.start_position()
    }
    fn end_position(&self) -> Position {
        self.update.end_position()
    }
}

/// `Expr` `Record`
#[derive(Debug, Clone)]
pub struct RecordUpdate {
    pub record: Expr,
    pub update: Record,
}
impl ParseTail for RecordUpdate {
    type Head = Expr;
    fn parse_tail<T: TokenRead>(parser: &mut Parser<T>, head: Self::Head) -> Result<Self> {
        Ok(RecordUpdate {
            record: head,
            update: track!(parser.parse())?,
        })
    }
}
impl PositionRange for RecordUpdate {
    fn start_position(&self) -> Position {
        self.record.start_position()
    }
    fn end_position(&self) -> Position {
        self.update.end_position()
    }
}

/// `try` `Body` `Option<TryOf>` `Option<TryCatch>` `Option<TryAfter>` `end`
#[derive(Debug, Clone)]
pub struct Try {
    pub _try: KeywordToken,
    pub body: Body,
    pub branch: Option<TryOf>,
    pub catch: Option<TryCatch>,
    pub after: Option<TryAfter>,
    pub _end: KeywordToken,
}
impl Parse for Try {
    fn parse<T: TokenRead>(parser: &mut Parser<T>) -> Result<Self> {
        Ok(Try {
            _try: track!(parser.expect(&Keyword::Try))?,
            body: track!(parser.parse())?,
            branch: track!(parser.parse())?,
            catch: track!(parser.parse())?,
            after: track!(parser.parse())?,
            _end: track!(parser.expect(&Keyword::End))?,
        })
    }
}
impl PositionRange for Try {
    fn start_position(&self) -> Position {
        self._try.start_position()
    }
    fn end_position(&self) -> Position {
        self._end.end_position()
    }
}

/// `receive` `Clauses<CaseClause>` `Option<Timeout>` `end`
#[derive(Debug, Clone)]
pub struct Receive {
    pub _receive: KeywordToken,
    pub clauses: Clauses<CaseClause>,
    pub timeout: Option<Timeout>,
    pub _end: KeywordToken,
}
impl Parse for Receive {
    fn parse<T: TokenRead>(parser: &mut Parser<T>) -> Result<Self> {
        Ok(Receive {
            _receive: track!(parser.expect(&Keyword::Receive))?,
            clauses: track!(parser.parse())?,
            timeout: track!(parser.parse())?,
            _end: track!(parser.expect(&Keyword::End))?,
        })
    }
}
impl PositionRange for Receive {
    fn start_position(&self) -> Position {
        self._receive.start_position()
    }
    fn end_position(&self) -> Position {
        self._end.end_position()
    }
}

/// `if` `Clauses<IfClause>` `end`
#[derive(Debug, Clone)]
pub struct If {
    pub _if: KeywordToken,
    pub clauses: Clauses<IfClause>,
    pub _end: KeywordToken,
}
impl Parse for If {
    fn parse<T: TokenRead>(parser: &mut Parser<T>) -> Result<Self> {
        Ok(If {
            _if: track!(parser.expect(&Keyword::If))?,
            clauses: track!(parser.parse())?,
            _end: track!(parser.expect(&Keyword::End))?,
        })
    }
}
impl PositionRange for If {
    fn start_position(&self) -> Position {
        self._if.start_position()
    }
    fn end_position(&self) -> Position {
        self._end.end_position()
    }
}

/// `case` `Expr` `of` `Clauses<CaseClause>` `end`
#[derive(Debug, Clone)]
pub struct Case {
    pub _case: KeywordToken,
    pub expr: Expr,
    pub _of: KeywordToken,
    pub clauses: Clauses<CaseClause>,
    pub _end: KeywordToken,
}
impl Parse for Case {
    fn parse<T: TokenRead>(parser: &mut Parser<T>) -> Result<Self> {
        Ok(Case {
            _case: track!(parser.expect(&Keyword::Case))?,
            expr: track!(parser.parse())?,
            _of: track!(parser.expect(&Keyword::Of))?,
            clauses: track!(parser.parse())?,
            _end: track!(parser.expect(&Keyword::End))?,
        })
    }
}
impl PositionRange for Case {
    fn start_position(&self) -> Position {
        self._case.start_position()
    }
    fn end_position(&self) -> Position {
        self._end.end_position()
    }
}

/// `DefinedFun | AnonymousFun | NamedFun`
#[derive(Debug, Clone)]
#[cfg_attr(feature = "cargo-clippy", allow(large_enum_variant))]
pub enum Fun {
    Defined(DefinedFun),
    Anonymous(AnonymousFun),
    Named(NamedFun),
}
impl Parse for Fun {
    fn parse<T: TokenRead>(parser: &mut Parser<T>) -> Result<Self> {
        // TODO: look ahead
        if let Ok(x) = parser.transaction(|parser| parser.parse()) {
            Ok(Fun::Defined(x))
        } else if let Ok(x) = parser.transaction(|parser| parser.parse()) {
            Ok(Fun::Anonymous(x))
        } else {
            Ok(Fun::Named(track!(parser.parse())?))
        }
    }
}
impl PositionRange for Fun {
    fn start_position(&self) -> Position {
        match *self {
            Fun::Defined(ref x) => x.start_position(),
            Fun::Anonymous(ref x) => x.start_position(),
            Fun::Named(ref x) => x.start_position(),
        }
    }
    fn end_position(&self) -> Position {
        match *self {
            Fun::Defined(ref x) => x.end_position(),
            Fun::Anonymous(ref x) => x.end_position(),
            Fun::Named(ref x) => x.end_position(),
        }
    }
}

/// `fun` `Option<ModulePrefix>` `NameAndArity`
#[derive(Debug, Clone)]
pub struct DefinedFun {
    pub _fun: KeywordToken,
    pub module: Option<ModulePrefix<AtomOrVariable>>,
    pub fun: NameAndArity<AtomOrVariable, IntegerOrVariable>,
}
impl Parse for DefinedFun {
    fn parse<T: TokenRead>(parser: &mut Parser<T>) -> Result<Self> {
        Ok(DefinedFun {
            _fun: track!(parser.expect(&Keyword::Fun))?,
            module: track!(parser.parse())?,
            fun: track!(parser.parse())?,
        })
    }
}
impl PositionRange for DefinedFun {
    fn start_position(&self) -> Position {
        self._fun.start_position()
    }
    fn end_position(&self) -> Position {
        self.fun.end_position()
    }
}

/// `fun` `Clauses<FunClause>` `end`
#[derive(Debug, Clone)]
pub struct AnonymousFun {
    pub _fun: KeywordToken,
    pub clauses: Clauses<FunClause>,
    pub _end: KeywordToken,
}
impl Parse for AnonymousFun {
    fn parse<T: TokenRead>(parser: &mut Parser<T>) -> Result<Self> {
        Ok(AnonymousFun {
            _fun: track!(parser.expect(&Keyword::Fun))?,
            clauses: track!(parser.parse())?,
            _end: track!(parser.expect(&Keyword::End))?,
        })
    }
}
impl PositionRange for AnonymousFun {
    fn start_position(&self) -> Position {
        self._fun.start_position()
    }
    fn end_position(&self) -> Position {
        self._end.end_position()
    }
}

/// `fun` `Clauses<NamedFunClause>` `end`
#[derive(Debug, Clone)]
pub struct NamedFun {
    pub _fun: KeywordToken,
    pub clauses: Clauses<NamedFunClause>,
    pub _end: KeywordToken,
}
impl Parse for NamedFun {
    fn parse<T: TokenRead>(parser: &mut Parser<T>) -> Result<Self> {
        Ok(NamedFun {
            _fun: track!(parser.expect(&Keyword::Fun))?,
            clauses: track!(parser.parse())?,
            _end: track!(parser.expect(&Keyword::End))?,
        })
    }
}
impl PositionRange for NamedFun {
    fn start_position(&self) -> Position {
        self._fun.start_position()
    }
    fn end_position(&self) -> Position {
        self._end.end_position()
    }
}

/// `[` `Expr` `||` `Sequence<Qualifier>` `]`
#[derive(Debug, Clone)]
pub struct ListComprehension {
    pub _open: SymbolToken,
    pub element: Expr,
    pub _bar: SymbolToken,
    pub qualifiers: Sequence<Qualifier>,
    pub _close: SymbolToken,
}
impl Parse for ListComprehension {
    fn parse<T: TokenRead>(parser: &mut Parser<T>) -> Result<Self> {
        Ok(ListComprehension {
            _open: track!(parser.expect(&Symbol::OpenSquare))?,
            element: track!(parser.parse())?,
            _bar: track!(parser.expect(&Symbol::DoubleVerticalBar))?,
            qualifiers: track!(parser.parse())?,
            _close: track!(parser.expect(&Symbol::CloseSquare))?,
        })
    }
}
impl PositionRange for ListComprehension {
    fn start_position(&self) -> Position {
        self._open.start_position()
    }
    fn end_position(&self) -> Position {
        self._close.end_position()
    }
}

/// `<<` `Expr` `||` `Sequence<Qualifiers>` `>>`
#[derive(Debug, Clone)]
pub struct BitsComprehension {
    pub _open: SymbolToken,
    pub element: Expr,
    pub _bar: SymbolToken,
    pub qualifiers: Sequence<Qualifier>,
    pub _close: SymbolToken,
}
impl Parse for BitsComprehension {
    fn parse<T: TokenRead>(parser: &mut Parser<T>) -> Result<Self> {
        Ok(BitsComprehension {
            _open: track!(parser.expect(&Symbol::DoubleLeftAngle))?,
            element: track!(parser.parse())?,
            _bar: track!(parser.expect(&Symbol::DoubleVerticalBar))?,
            qualifiers: track!(parser.parse())?,
            _close: track!(parser.expect(&Symbol::DoubleRightAngle))?,
        })
    }
}
impl PositionRange for BitsComprehension {
    fn start_position(&self) -> Position {
        self._open.start_position()
    }
    fn end_position(&self) -> Position {
        self._close.end_position()
    }
}

/// `catch` `Body`
#[derive(Debug, Clone)]
pub struct Catch {
    pub _catch: KeywordToken,
    pub expr: Body,
}
impl Parse for Catch {
    fn parse<T: TokenRead>(parser: &mut Parser<T>) -> Result<Self> {
        Ok(Catch {
            _catch: track!(parser.expect(&Keyword::Catch))?,
            expr: track!(parser.parse())?,
        })
    }
}
impl PositionRange for Catch {
    fn start_position(&self) -> Position {
        self._catch.start_position()
    }
    fn end_position(&self) -> Position {
        self.expr.end_position()
    }
}

/// `begin` `Body` `end`
#[derive(Debug, Clone)]
pub struct Block {
    pub _begin: KeywordToken,
    pub body: Body,
    pub _end: KeywordToken,
}
impl Parse for Block {
    fn parse<T: TokenRead>(parser: &mut Parser<T>) -> Result<Self> {
        Ok(Block {
            _begin: track!(parser.expect(&Keyword::Begin))?,
            body: track!(parser.parse())?,
            _end: track!(parser.expect(&Keyword::End))?,
        })
    }
}
impl PositionRange for Block {
    fn start_position(&self) -> Position {
        self._begin.start_position()
    }
    fn end_position(&self) -> Position {
        self._end.end_position()
    }
}

/// `Expr` `RecordFieldIndex`
#[derive(Debug, Clone)]
pub struct RecordFieldAccess<T = Expr> {
    pub record: T,
    pub index: RecordFieldIndex,
}
impl<T> ParseTail for RecordFieldAccess<T> {
    type Head = T;
    fn parse_tail<U: TokenRead>(parser: &mut Parser<U>, head: Self::Head) -> Result<Self> {
        Ok(RecordFieldAccess {
            record: head,
            index: track!(parser.parse())?,
        })
    }
}
impl<T: PositionRange> PositionRange for RecordFieldAccess<T> {
    fn start_position(&self) -> Position {
        self.record.start_position()
    }
    fn end_position(&self) -> Position {
        self.index.end_position()
    }
}
