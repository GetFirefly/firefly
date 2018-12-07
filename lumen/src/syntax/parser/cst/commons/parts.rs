use std::marker::PhantomData;

use trackable::{track, track_panic};

use crate::syntax::tokenizer::tokens::{AtomToken, IntegerToken, KeywordToken, SymbolToken};
use crate::syntax::tokenizer::values::{Keyword, Symbol};
use crate::syntax::tokenizer::{LexicalToken, Position, PositionRange};

use crate::syntax::parser::traits::{Delimiter, Parse, TokenRead};
use crate::syntax::parser::{ErrorKind, Parser, Result};

use super::iterators::{ConsCellIter, SequenceIter};
use super::AtomOrVariable;

pub type Clauses<T> = Sequence<T, Semicolon>;

/// `T` `Option<BitsElemSize<T>>` `Option<BitsElemSpecs>`
#[derive(Debug, Clone)]
pub struct BitsElem<T> {
    pub element: T,
    pub size: Option<BitsElemSize<T>>,
    pub type_specs: Option<BitsElemSpecs>,
}
impl<T: Parse> Parse for BitsElem<T> {
    fn parse<U: TokenRead>(parser: &mut Parser<U>) -> Result<Self> {
        Ok(BitsElem {
            element: track!(T::parse_non_left_recor(parser))?,
            size: track!(parser.parse())?,
            type_specs: track!(parser.parse())?,
        })
    }
}
impl<T: PositionRange> PositionRange for BitsElem<T> {
    fn start_position(&self) -> Position {
        self.element.start_position()
    }
    fn end_position(&self) -> Position {
        self.type_specs
            .as_ref()
            .map(|t| t.end_position())
            .or_else(|| self.size.as_ref().map(|t| t.end_position()))
            .unwrap_or_else(|| self.element.end_position())
    }
}

/// `:` `T`
#[derive(Debug, Clone)]
pub struct BitsElemSize<T> {
    pub _colon: SymbolToken,
    pub size: T,
}
impl<T: Parse> Parse for BitsElemSize<T> {
    fn parse<U: TokenRead>(parser: &mut Parser<U>) -> Result<Self> {
        Ok(BitsElemSize {
            _colon: track!(parser.expect(&Symbol::Colon))?,
            size: track!(T::parse_non_left_recor(parser))?,
        })
    }
}
impl<T: PositionRange> PositionRange for BitsElemSize<T> {
    fn start_position(&self) -> Position {
        self._colon.start_position()
    }
    fn end_position(&self) -> Position {
        self.size.end_position()
    }
}

/// `/` `Sequence<BitsElemSpec, Hyphen>`
#[derive(Debug, Clone)]
pub struct BitsElemSpecs {
    pub _slash: SymbolToken,
    pub specs: Sequence<BitsElemSpec, Hyphen>,
}
impl Parse for BitsElemSpecs {
    fn parse<T: TokenRead>(parser: &mut Parser<T>) -> Result<Self> {
        Ok(BitsElemSpecs {
            _slash: track!(parser.expect(&Symbol::Slash))?,
            specs: track!(parser.parse())?,
        })
    }
}
impl PositionRange for BitsElemSpecs {
    fn start_position(&self) -> Position {
        self._slash.start_position()
    }
    fn end_position(&self) -> Position {
        self.specs.end_position()
    }
}

/// `AtomToken` | (`unit` `:` `IntegerToken`)
#[derive(Debug, Clone)]
pub enum BitsElemSpec {
    Type(AtomToken),
    Unit {
        _unit: AtomToken,
        _colon: SymbolToken,
        unit: IntegerToken,
    },
}
impl Parse for BitsElemSpec {
    fn parse<T: TokenRead>(parser: &mut Parser<T>) -> Result<Self> {
        let atom: AtomToken = track!(parser.parse())?;
        if atom.value() == "unit" {
            Ok(BitsElemSpec::Unit {
                _unit: atom,
                _colon: track!(parser.expect(&Symbol::Colon))?,
                unit: track!(parser.parse())?,
            })
        } else {
            Ok(BitsElemSpec::Type(atom))
        }
    }
}
impl PositionRange for BitsElemSpec {
    fn start_position(&self) -> Position {
        match *self {
            BitsElemSpec::Type(ref t) => t.start_position(),
            BitsElemSpec::Unit { ref _unit, .. } => _unit.start_position(),
        }
    }
    fn end_position(&self) -> Position {
        match *self {
            BitsElemSpec::Type(ref t) => t.end_position(),
            BitsElemSpec::Unit { ref unit, .. } => unit.end_position(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ConsCell<T> {
    pub item: T,
    pub tail: Option<ConsCellTail<T>>,
}
impl<T> ConsCell<T> {
    pub fn iter(&self) -> ConsCellIter<T> {
        ConsCellIter::new(self)
    }
}
impl<T: Parse> Parse for ConsCell<T> {
    fn parse<U: TokenRead>(parser: &mut Parser<U>) -> Result<Self> {
        Ok(ConsCell {
            item: track!(parser.parse())?,
            tail: track!(parser.parse())?,
        })
    }
}
impl<T: PositionRange> PositionRange for ConsCell<T> {
    fn start_position(&self) -> Position {
        self.item.start_position()
    }
    fn end_position(&self) -> Position {
        self.tail
            .as_ref()
            .map(|t| t.end_position())
            .unwrap_or_else(|| self.item.end_position())
    }
}

#[derive(Debug, Clone)]
pub enum ConsCellTail<T> {
    Proper {
        _comma: SymbolToken,
        item: T,
        tail: Option<Box<ConsCellTail<T>>>,
    },
    Improper {
        _bar: SymbolToken,
        item: T,
    },
}
impl<T: Parse> Parse for ConsCellTail<T> {
    fn parse<U: TokenRead>(parser: &mut Parser<U>) -> Result<Self> {
        let symbol: SymbolToken = track!(parser.parse())?;
        match symbol.value() {
            Symbol::Comma => Ok(ConsCellTail::Proper {
                _comma: symbol,
                item: track!(parser.parse())?,
                tail: track!(parser.parse())?,
            }),
            Symbol::VerticalBar => Ok(ConsCellTail::Improper {
                _bar: symbol,
                item: track!(parser.parse())?,
            }),
            _ => {
                track_panic!(ErrorKind::InvalidInput, "Unexpected symbol: {:?}", symbol);
            }
        }
    }
}
impl<T: PositionRange> PositionRange for ConsCellTail<T> {
    fn start_position(&self) -> Position {
        match *self {
            ConsCellTail::Proper { ref _comma, .. } => _comma.start_position(),
            ConsCellTail::Improper { ref _bar, .. } => _bar.start_position(),
        }
    }
    fn end_position(&self) -> Position {
        match *self {
            ConsCellTail::Proper {
                ref item, ref tail, ..
            } => tail
                .as_ref()
                .map(|t| t.end_position())
                .unwrap_or_else(|| item.end_position()),
            ConsCellTail::Improper { ref item, .. } => item.end_position(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum BinaryOp {
    Plus(SymbolToken),
    Minus(SymbolToken),
    Mul(SymbolToken),
    FloatDiv(SymbolToken),
    IntDiv(KeywordToken),
    Rem(KeywordToken),
    Bor(KeywordToken),
    Bxor(KeywordToken),
    Band(KeywordToken),
    Bsl(KeywordToken),
    Bsr(KeywordToken),
    Or(KeywordToken),
    Xor(KeywordToken),
    PlusPlus(SymbolToken),
    MinusMinus(SymbolToken),
    Eq(SymbolToken),
    ExactEq(SymbolToken),
    NotEq(SymbolToken),
    ExactNotEq(SymbolToken),
    Less(SymbolToken),
    LessEq(SymbolToken),
    Greater(SymbolToken),
    GreaterEq(SymbolToken),
    Andalso(KeywordToken),
    Orelse(KeywordToken),
    Send(SymbolToken),
}
impl BinaryOp {
    pub fn from_token(token: LexicalToken) -> ::std::result::Result<Self, LexicalToken> {
        match token {
            LexicalToken::Symbol(s) => match s.value() {
                Symbol::Plus => Ok(BinaryOp::Plus(s)),
                Symbol::Hyphen => Ok(BinaryOp::Minus(s)),
                Symbol::Multiply => Ok(BinaryOp::Mul(s)),
                Symbol::Slash => Ok(BinaryOp::FloatDiv(s)),
                Symbol::PlusPlus => Ok(BinaryOp::PlusPlus(s)),
                Symbol::MinusMinus => Ok(BinaryOp::MinusMinus(s)),
                Symbol::Eq => Ok(BinaryOp::Eq(s)),
                Symbol::ExactEq => Ok(BinaryOp::ExactEq(s)),
                Symbol::NotEq => Ok(BinaryOp::NotEq(s)),
                Symbol::ExactNotEq => Ok(BinaryOp::ExactNotEq(s)),
                Symbol::Less => Ok(BinaryOp::Less(s)),
                Symbol::LessEq => Ok(BinaryOp::LessEq(s)),
                Symbol::Greater => Ok(BinaryOp::Greater(s)),
                Symbol::GreaterEq => Ok(BinaryOp::GreaterEq(s)),
                Symbol::Not => Ok(BinaryOp::Send(s)),
                _ => Err(s.into()),
            },
            LexicalToken::Keyword(k) => match k.value() {
                Keyword::Div => Ok(BinaryOp::IntDiv(k)),
                Keyword::Rem => Ok(BinaryOp::Rem(k)),
                Keyword::Bor => Ok(BinaryOp::Bor(k)),
                Keyword::Bxor => Ok(BinaryOp::Bxor(k)),
                Keyword::Band => Ok(BinaryOp::Band(k)),
                Keyword::Bsl => Ok(BinaryOp::Bsl(k)),
                Keyword::Bsr => Ok(BinaryOp::Bsr(k)),
                Keyword::Or => Ok(BinaryOp::Or(k)),
                Keyword::Xor => Ok(BinaryOp::Xor(k)),
                Keyword::Andalso => Ok(BinaryOp::Andalso(k)),
                Keyword::Orelse => Ok(BinaryOp::Orelse(k)),
                _ => Err(k.into()),
            },
            _ => Err(token),
        }
    }
}
impl Parse for BinaryOp {
    fn parse<T: TokenRead>(parser: &mut Parser<T>) -> Result<Self> {
        let token = track!(parser.parse::<LexicalToken>())?;
        match Self::from_token(token) {
            Err(token) => track_panic!(ErrorKind::UnexpectedToken(token)),
            Ok(op) => Ok(op),
        }
    }
}
impl PositionRange for BinaryOp {
    fn start_position(&self) -> Position {
        match *self {
            BinaryOp::IntDiv(ref t)
            | BinaryOp::Rem(ref t)
            | BinaryOp::Bor(ref t)
            | BinaryOp::Bxor(ref t)
            | BinaryOp::Band(ref t)
            | BinaryOp::Bsl(ref t)
            | BinaryOp::Bsr(ref t)
            | BinaryOp::Or(ref t)
            | BinaryOp::Xor(ref t)
            | BinaryOp::Andalso(ref t)
            | BinaryOp::Orelse(ref t) => t.start_position(),
            BinaryOp::Plus(ref t)
            | BinaryOp::Minus(ref t)
            | BinaryOp::Mul(ref t)
            | BinaryOp::FloatDiv(ref t)
            | BinaryOp::PlusPlus(ref t)
            | BinaryOp::MinusMinus(ref t)
            | BinaryOp::Eq(ref t)
            | BinaryOp::ExactEq(ref t)
            | BinaryOp::NotEq(ref t)
            | BinaryOp::ExactNotEq(ref t)
            | BinaryOp::Less(ref t)
            | BinaryOp::LessEq(ref t)
            | BinaryOp::Greater(ref t)
            | BinaryOp::GreaterEq(ref t)
            | BinaryOp::Send(ref t) => t.start_position(),
        }
    }
    fn end_position(&self) -> Position {
        match *self {
            BinaryOp::IntDiv(ref t)
            | BinaryOp::Rem(ref t)
            | BinaryOp::Bor(ref t)
            | BinaryOp::Bxor(ref t)
            | BinaryOp::Band(ref t)
            | BinaryOp::Bsl(ref t)
            | BinaryOp::Bsr(ref t)
            | BinaryOp::Or(ref t)
            | BinaryOp::Xor(ref t)
            | BinaryOp::Andalso(ref t)
            | BinaryOp::Orelse(ref t) => t.end_position(),
            BinaryOp::Plus(ref t)
            | BinaryOp::Minus(ref t)
            | BinaryOp::Mul(ref t)
            | BinaryOp::FloatDiv(ref t)
            | BinaryOp::PlusPlus(ref t)
            | BinaryOp::MinusMinus(ref t)
            | BinaryOp::Eq(ref t)
            | BinaryOp::ExactEq(ref t)
            | BinaryOp::NotEq(ref t)
            | BinaryOp::ExactNotEq(ref t)
            | BinaryOp::Less(ref t)
            | BinaryOp::LessEq(ref t)
            | BinaryOp::Greater(ref t)
            | BinaryOp::GreaterEq(ref t)
            | BinaryOp::Send(ref t) => t.end_position(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum UnaryOp {
    Plus(SymbolToken),
    Minus(SymbolToken),
    Not(KeywordToken),
    Bnot(KeywordToken),
}
impl UnaryOp {
    pub fn from_token(token: LexicalToken) -> ::std::result::Result<Self, LexicalToken> {
        match token {
            LexicalToken::Symbol(s) => match s.value() {
                Symbol::Plus => Ok(UnaryOp::Plus(s)),
                Symbol::Hyphen => Ok(UnaryOp::Minus(s)),
                _ => Err(s.into()),
            },
            LexicalToken::Keyword(k) => match k.value() {
                Keyword::Not => Ok(UnaryOp::Not(k)),
                Keyword::Bnot => Ok(UnaryOp::Bnot(k)),
                _ => Err(k.into()),
            },
            token => Err(token),
        }
    }
}
impl Parse for UnaryOp {
    fn parse<T: TokenRead>(parser: &mut Parser<T>) -> Result<Self> {
        let token = track!(parser.parse())?;
        match UnaryOp::from_token(token) {
            Err(token) => track_panic!(ErrorKind::UnexpectedToken(token)),
            Ok(op) => Ok(op),
        }
    }
}
impl PositionRange for UnaryOp {
    fn start_position(&self) -> Position {
        match *self {
            UnaryOp::Plus(ref t) | UnaryOp::Minus(ref t) => t.start_position(),
            UnaryOp::Not(ref t) | UnaryOp::Bnot(ref t) => t.start_position(),
        }
    }
    fn end_position(&self) -> Position {
        match *self {
            UnaryOp::Plus(ref t) | UnaryOp::Minus(ref t) => t.end_position(),
            UnaryOp::Not(ref t) | UnaryOp::Bnot(ref t) => t.end_position(),
        }
    }
}

/// `(` `Option<Sequence<T>>` `)`
#[derive(Debug, Clone)]
pub struct Args<T> {
    pub _open: SymbolToken,
    pub args: Option<Sequence<T>>,
    pub _close: SymbolToken,
}
impl<T: Parse> Parse for Args<T> {
    fn parse<U: TokenRead>(parser: &mut Parser<U>) -> Result<Self>
    where
        U: TokenRead,
    {
        Ok(Args {
            _open: track!(parser.expect(&Symbol::OpenParen))?,
            args: track!(parser.parse())?,
            _close: track!(parser.expect(&Symbol::CloseParen))?,
        })
    }
}
impl<T> PositionRange for Args<T> {
    fn start_position(&self) -> Position {
        self._open.start_position()
    }
    fn end_position(&self) -> Position {
        self._close.end_position()
    }
}

/// `T` `:`
#[derive(Debug, Clone)]
pub struct ModulePrefix<T> {
    pub name: T,
    pub _colon: SymbolToken,
}
impl<T: Parse> Parse for ModulePrefix<T> {
    fn parse<U: TokenRead>(parser: &mut Parser<U>) -> Result<Self> {
        Ok(ModulePrefix {
            name: track!(T::parse_non_left_recor(parser))?,
            _colon: track!(parser.expect(&Symbol::Colon))?,
        })
    }
}
impl<T: PositionRange> PositionRange for ModulePrefix<T> {
    fn start_position(&self) -> Position {
        self.name.start_position()
    }
    fn end_position(&self) -> Position {
        self._colon.end_position()
    }
}

/// `T` (`:=`|`=>`) `T`
#[derive(Debug, Clone)]
pub struct MapField<T> {
    pub key: T,
    pub _relation: SymbolToken,
    pub value: T,
}
impl<T: Parse> Parse for MapField<T> {
    fn parse<U: TokenRead>(parser: &mut Parser<U>) -> Result<Self> {
        Ok(MapField {
            key: track!(parser.parse())?,
            _relation: track!(parser.expect_any(&[&Symbol::DoubleRightArrow, &Symbol::MapMatch],))?,
            value: track!(parser.parse())?,
        })
    }
}
impl<T: PositionRange> PositionRange for MapField<T> {
    fn start_position(&self) -> Position {
        self.key.start_position()
    }
    fn end_position(&self) -> Position {
        self.value.end_position()
    }
}

/// `AtomOrVariable` `=` `T`
#[derive(Debug, Clone)]
pub struct RecordField<T> {
    pub key: AtomOrVariable,
    pub _bind: SymbolToken,
    pub value: T,
}
impl<T: Parse> Parse for RecordField<T> {
    fn parse<U: TokenRead>(parser: &mut Parser<U>) -> Result<Self> {
        Ok(RecordField {
            key: track!(parser.parse())?,
            _bind: track!(parser.parse())?,
            value: track!(parser.parse())?,
        })
    }
}
impl<T: PositionRange> PositionRange for RecordField<T> {
    fn start_position(&self) -> Position {
        self.key.start_position()
    }
    fn end_position(&self) -> Position {
        self.value.end_position()
    }
}

/// `AtomToken` `/` `IntegerToken`
#[derive(Debug, Clone)]
pub struct NameAndArity<N = AtomToken, A = IntegerToken> {
    pub name: N,
    pub _slash: SymbolToken,
    pub arity: A,
}
impl<N: Parse, A: Parse> Parse for NameAndArity<N, A> {
    fn parse<T: TokenRead>(parser: &mut Parser<T>) -> Result<Self> {
        Ok(NameAndArity {
            name: track!(parser.parse())?,
            _slash: track!(parser.expect(&Symbol::Slash))?,
            arity: track!(parser.parse())?,
        })
    }
}
impl<N: PositionRange, A: PositionRange> PositionRange for NameAndArity<N, A> {
    fn start_position(&self) -> Position {
        self.name.start_position()
    }
    fn end_position(&self) -> Position {
        self.arity.end_position()
    }
}

/// `item` `Option<SequenceTail<T>>`
#[derive(Debug, Clone)]
pub struct Sequence<T, D = Comma> {
    pub item: T,
    pub tail: Option<SequenceTail<T, D>>,
}
impl<T, D> Sequence<T, D> {
    pub fn iter(&self) -> SequenceIter<T, D> {
        SequenceIter::new(self)
    }
}
impl<T: Parse, D: Delimiter> Parse for Sequence<T, D> {
    fn parse<U>(parser: &mut Parser<U>) -> Result<Self>
    where
        U: TokenRead,
    {
        Ok(Sequence {
            item: track!(parser.parse())?,
            tail: track!(parser.parse())?,
        })
    }
}
impl<T: PositionRange, D> PositionRange for Sequence<T, D> {
    fn start_position(&self) -> Position {
        self.item.start_position()
    }
    fn end_position(&self) -> Position {
        self.tail
            .as_ref()
            .map(|t| t.end_position())
            .unwrap_or_else(|| self.item.end_position())
    }
}

/// `,` `item` `Option<SequenceTail<T>>`
#[derive(Debug, Clone)]
pub struct SequenceTail<T, D> {
    pub _comma: SymbolToken,
    pub item: T,
    pub tail: Option<Box<SequenceTail<T, D>>>,
    _phantom: PhantomData<D>,
}
impl<T: Parse, D: Delimiter> Parse for SequenceTail<T, D> {
    fn parse<U>(parser: &mut Parser<U>) -> Result<Self>
    where
        U: TokenRead,
    {
        Ok(SequenceTail {
            _comma: track!(parser.expect(&D::delimiter()))?,
            item: track!(parser.parse())?,
            tail: track!(parser.parse())?,
            _phantom: PhantomData,
        })
    }
}
impl<T: PositionRange, D> PositionRange for SequenceTail<T, D> {
    fn start_position(&self) -> Position {
        self._comma.start_position()
    }
    fn end_position(&self) -> Position {
        self.tail
            .as_ref()
            .map(|t| t.end_position())
            .unwrap_or_else(|| self.item.end_position())
    }
}

#[derive(Debug, Clone)]
pub struct Comma;
impl Delimiter for Comma {
    fn delimiter() -> Symbol {
        Symbol::Comma
    }
}

#[derive(Debug, Clone)]
pub struct Hyphen;
impl Delimiter for Hyphen {
    fn delimiter() -> Symbol {
        Symbol::Hyphen
    }
}

#[derive(Debug, Clone)]
pub struct Semicolon;
impl Delimiter for Semicolon {
    fn delimiter() -> Symbol {
        Symbol::Semicolon
    }
}
