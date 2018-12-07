pub mod parts;

use trackable::track;

use crate::syntax::tokenizer::tokens::{AtomToken, KeywordToken, SymbolToken, VariableToken};
use crate::syntax::tokenizer::values::{Keyword, Symbol};
use crate::syntax::tokenizer::{Position, PositionRange};

use crate::syntax::parser::traits::{Parse, ParseTail, TokenRead};
use crate::syntax::parser::{Parser, Result};

use super::commons;
use super::commons::parts::{Args, Sequence};
use super::Type;

use self::parts::{BitsSpec, ListElement};

pub type Tuple = commons::Tuple<Type>;
pub type Map = commons::Map<Type>;
pub type Record = commons::Record<Type>;
pub type Parenthesized = commons::Parenthesized<Type>;
pub type TypeCall = commons::Call<AtomToken, Type>;
pub type UnaryOpCall = commons::UnaryOpCall<Type>;
pub type BinaryOpCall = commons::BinaryOpCall<Type>;

/// `AnyFun | AnyArityFun | NormalFun`
#[derive(Debug, Clone)]
#[cfg_attr(feature = "cargo-clippy", allow(large_enum_variant))]
pub enum Fun {
    Any(AnyFun),
    AnyArity(AnyArityFun),
    Normal(NormalFun),
}
impl Parse for Fun {
    fn parse<T>(parser: &mut Parser<T>) -> Result<Self>
    where
        T: TokenRead,
    {
        // TODO: look ahead
        if let Ok(x) = parser.transaction(|parser| parser.parse()) {
            Ok(Fun::Any(x))
        } else if let Ok(x) = parser.transaction(|parser| parser.parse()) {
            Ok(Fun::AnyArity(x))
        } else {
            Ok(Fun::Normal(track!(parser.parse())?))
        }
    }
}
impl PositionRange for Fun {
    fn start_position(&self) -> Position {
        match *self {
            Fun::Any(ref x) => x.start_position(),
            Fun::AnyArity(ref x) => x.start_position(),
            Fun::Normal(ref x) => x.start_position(),
        }
    }
    fn end_position(&self) -> Position {
        match *self {
            Fun::Any(ref x) => x.end_position(),
            Fun::AnyArity(ref x) => x.end_position(),
            Fun::Normal(ref x) => x.end_position(),
        }
    }
}

/// `fun` `(` `)`
#[derive(Debug, Clone)]
pub struct AnyFun {
    pub _fun: KeywordToken,
    pub _open: SymbolToken,
    pub _close: SymbolToken,
}
impl Parse for AnyFun {
    fn parse<T>(parser: &mut Parser<T>) -> Result<Self>
    where
        T: TokenRead,
    {
        Ok(AnyFun {
            _fun: track!(parser.expect(&Keyword::Fun))?,
            _open: track!(parser.expect(&Symbol::OpenParen))?,
            _close: track!(parser.expect(&Symbol::CloseParen))?,
        })
    }
}
impl PositionRange for AnyFun {
    fn start_position(&self) -> Position {
        self._fun.start_position()
    }
    fn end_position(&self) -> Position {
        self._close.end_position()
    }
}

/// `fun` `(` `(` `...` `)` `)` `->` `Type` `)`
#[derive(Debug, Clone)]
pub struct AnyArityFun {
    pub _fun: KeywordToken,
    pub _open: SymbolToken,
    pub _args_open: SymbolToken,
    pub _args: SymbolToken,
    pub _args_close: SymbolToken,
    pub _arrow: SymbolToken,
    pub return_type: Type,
    pub _close: SymbolToken,
}
impl Parse for AnyArityFun {
    fn parse<T>(parser: &mut Parser<T>) -> Result<Self>
    where
        T: TokenRead,
    {
        Ok(AnyArityFun {
            _fun: track!(parser.expect(&Keyword::Fun))?,
            _open: track!(parser.expect(&Symbol::OpenParen))?,
            _args_open: track!(parser.expect(&Symbol::OpenParen))?,
            _args: track!(parser.expect(&Symbol::TripleDot))?,
            _args_close: track!(parser.expect(&Symbol::CloseParen))?,
            _arrow: track!(parser.expect(&Symbol::RightArrow))?,
            return_type: track!(parser.parse())?,
            _close: track!(parser.expect(&Symbol::CloseParen))?,
        })
    }
}
impl PositionRange for AnyArityFun {
    fn start_position(&self) -> Position {
        self._fun.start_position()
    }
    fn end_position(&self) -> Position {
        self._close.end_position()
    }
}

/// `fun` `(` `Args<Type>` `->` `Type` `)`
#[derive(Debug, Clone)]
pub struct NormalFun {
    pub _fun: KeywordToken,
    pub _open: SymbolToken,
    pub args: Args<Type>,
    pub _arrow: SymbolToken,
    pub return_type: Type,
    pub _close: SymbolToken,
}
impl Parse for NormalFun {
    fn parse<T>(parser: &mut Parser<T>) -> Result<Self>
    where
        T: TokenRead,
    {
        Ok(NormalFun {
            _fun: track!(parser.expect(&Keyword::Fun))?,
            _open: track!(parser.expect(&Symbol::OpenParen))?,
            args: track!(parser.parse())?,
            _arrow: track!(parser.expect(&Symbol::RightArrow))?,
            return_type: track!(parser.parse())?,
            _close: track!(parser.expect(&Symbol::CloseParen))?,
        })
    }
}
impl PositionRange for NormalFun {
    fn start_position(&self) -> Position {
        self._fun.start_position()
    }
    fn end_position(&self) -> Position {
        self._close.end_position()
    }
}

/// `when` `Sequence<Type>`
#[derive(Debug, Clone)]
pub struct Constraints {
    pub _when: KeywordToken,
    pub constraints: Sequence<Type>,
}
impl Parse for Constraints {
    fn parse<T>(parser: &mut Parser<T>) -> Result<Self>
    where
        T: TokenRead,
    {
        Ok(Constraints {
            _when: track!(parser.expect(&Keyword::When))?,
            constraints: track!(parser.parse())?,
        })
    }
}
impl PositionRange for Constraints {
    fn start_position(&self) -> Position {
        self._when.start_position()
    }
    fn end_position(&self) -> Position {
        self.constraints.end_position()
    }
}

/// `Type` `..` `Type`
#[derive(Debug, Clone)]
pub struct Range {
    pub low: Type,
    pub _dot: SymbolToken,
    pub high: Type,
}
impl ParseTail for Range {
    type Head = Type;
    fn parse_tail<T>(parser: &mut Parser<T>, head: Self::Head) -> Result<Self>
    where
        T: TokenRead,
    {
        Ok(Range {
            low: head,
            _dot: track!(parser.expect(&Symbol::DoubleDot))?,
            high: track!(parser.parse())?,
        })
    }
}
impl PositionRange for Range {
    fn start_position(&self) -> Position {
        self.low.start_position()
    }
    fn end_position(&self) -> Position {
        self.high.end_position()
    }
}

/// `Type` `|` `Type`
#[derive(Debug, Clone)]
pub struct Union {
    pub left: Type,
    pub _or: SymbolToken,
    pub right: Type,
}
impl ParseTail for Union {
    type Head = Type;
    fn parse_tail<T>(parser: &mut Parser<T>, head: Self::Head) -> Result<Self>
    where
        T: TokenRead,
    {
        Ok(Union {
            left: head,
            _or: track!(parser.expect(&Symbol::VerticalBar))?,
            right: track!(parser.parse())?,
        })
    }
}
impl PositionRange for Union {
    fn start_position(&self) -> Position {
        self.left.start_position()
    }
    fn end_position(&self) -> Position {
        self.right.end_position()
    }
}

/// `VariableToken` `::` `Type`
#[derive(Debug, Clone)]
pub struct Annotated {
    pub var: VariableToken,
    pub _colon: SymbolToken,
    pub ty: Type,
}
impl Parse for Annotated {
    fn parse<T>(parser: &mut Parser<T>) -> Result<Self>
    where
        T: TokenRead,
    {
        Ok(Annotated {
            var: track!(parser.parse())?,
            _colon: track!(parser.expect(&Symbol::DoubleColon))?,
            ty: track!(parser.parse())?,
        })
    }
}
impl PositionRange for Annotated {
    fn start_position(&self) -> Position {
        self.var.start_position()
    }
    fn end_position(&self) -> Position {
        self.ty.end_position()
    }
}

/// `[` `Option<ListElement>` `]`
#[derive(Debug, Clone)]
pub struct List {
    pub _open: SymbolToken,
    pub element: Option<ListElement>,
    pub _close: SymbolToken,
}
impl Parse for List {
    fn parse<T>(parser: &mut Parser<T>) -> Result<Self>
    where
        T: TokenRead,
    {
        Ok(List {
            _open: track!(parser.expect(&Symbol::OpenSquare))?,
            element: track!(parser.parse())?,
            _close: track!(parser.expect(&Symbol::CloseSquare))?,
        })
    }
}
impl PositionRange for List {
    fn start_position(&self) -> Position {
        self._open.start_position()
    }
    fn end_position(&self) -> Position {
        self._close.end_position()
    }
}

/// `<<` `Option<BitsSpec>` `>>`
#[derive(Debug, Clone)]
pub struct Bits {
    pub _open: SymbolToken,
    pub spec: Option<BitsSpec>,
    pub _close: SymbolToken,
}
impl Parse for Bits {
    fn parse<T>(parser: &mut Parser<T>) -> Result<Self>
    where
        T: TokenRead,
    {
        Ok(Bits {
            _open: track!(parser.expect(&Symbol::DoubleLeftAngle))?,
            spec: track!(parser.parse())?,
            _close: track!(parser.expect(&Symbol::DoubleRightAngle))?,
        })
    }
}
impl PositionRange for Bits {
    fn start_position(&self) -> Position {
        self._open.start_position()
    }
    fn end_position(&self) -> Position {
        self._close.end_position()
    }
}
