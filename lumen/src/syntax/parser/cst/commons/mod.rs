use trackable::{track, track_panic};

use crate::syntax::tokenizer::tokens::{AtomToken, IntegerToken, SymbolToken, VariableToken};
use crate::syntax::tokenizer::values::Symbol;
use crate::syntax::tokenizer::{LexicalToken, Position, PositionRange};

use crate::syntax::parser::traits::{Parse, ParseTail, TokenRead};
use crate::syntax::parser::{ErrorKind, Parser, Result};

use self::parts::{Args, BinaryOp, BitsElem, ConsCell, MapField};
use self::parts::{ModulePrefix, RecordField, Sequence, UnaryOp};
use super::Pattern;

pub mod iterators;
pub mod parts;

/// `{` `Option<Sequence<T>>` `}`
#[derive(Debug, Clone)]
pub struct Tuple<T> {
    pub _open: SymbolToken,
    pub elements: Option<Sequence<T>>,
    pub _close: SymbolToken,
}
impl<T: Parse> Parse for Tuple<T> {
    fn parse<U: TokenRead>(parser: &mut Parser<U>) -> Result<Self> {
        Ok(Tuple {
            _open: track!(parser.expect(&Symbol::OpenBrace))?,
            elements: track!(parser.parse())?,
            _close: track!(parser.expect(&Symbol::CloseBrace))?,
        })
    }
}
impl<T> PositionRange for Tuple<T> {
    fn start_position(&self) -> Position {
        self._open.start_position()
    }
    fn end_position(&self) -> Position {
        self._close.end_position()
    }
}

/// `[` `Option<ConsCell<T>>` `]`
#[derive(Debug, Clone)]
pub struct List<T> {
    pub _open: SymbolToken,
    pub elements: Option<ConsCell<T>>,
    pub _close: SymbolToken,
}
impl<T: Parse> Parse for List<T> {
    fn parse<U: TokenRead>(parser: &mut Parser<U>) -> Result<Self> {
        Ok(List {
            _open: track!(parser.expect(&Symbol::OpenSquare))?,
            elements: track!(parser.parse())?,
            _close: track!(parser.expect(&Symbol::CloseSquare))?,
        })
    }
}
impl<T> PositionRange for List<T> {
    fn start_position(&self) -> Position {
        self._open.start_position()
    }
    fn end_position(&self) -> Position {
        self._close.end_position()
    }
}

/// `[` `Option<Sequence<T>>` `]`
#[derive(Debug, Clone)]
pub struct ProperList<T> {
    pub _open: SymbolToken,
    pub elements: Option<Sequence<T>>,
    pub _close: SymbolToken,
}
impl<T: Parse> Parse for ProperList<T> {
    fn parse<U: TokenRead>(parser: &mut Parser<U>) -> Result<Self> {
        Ok(ProperList {
            _open: track!(parser.expect(&Symbol::OpenSquare))?,
            elements: track!(parser.parse())?,
            _close: track!(parser.expect(&Symbol::CloseSquare))?,
        })
    }
}
impl<T> PositionRange for ProperList<T> {
    fn start_position(&self) -> Position {
        self._open.start_position()
    }
    fn end_position(&self) -> Position {
        self._close.end_position()
    }
}

/// `<<` `Option<Sequence<BitsElem<T>>>` `>>`
#[derive(Debug, Clone)]
pub struct Bits<T> {
    pub _open: SymbolToken,
    pub elements: Option<Sequence<BitsElem<T>>>,
    pub _close: SymbolToken,
}
impl<T: Parse> Parse for Bits<T> {
    fn parse<U: TokenRead>(parser: &mut Parser<U>) -> Result<Self> {
        Ok(Bits {
            _open: track!(parser.expect(&Symbol::DoubleLeftAngle))?,
            elements: track!(parser.parse())?,
            _close: track!(parser.expect(&Symbol::DoubleRightAngle))?,
        })
    }
}
impl<T> PositionRange for Bits<T> {
    fn start_position(&self) -> Position {
        self._open.start_position()
    }
    fn end_position(&self) -> Position {
        self._close.end_position()
    }
}

/// `#` `AtomToken` `{` `Option<Sequence<RecordField<T>>>` `}`
#[derive(Debug, Clone)]
pub struct Record<T> {
    pub _sharp: SymbolToken,
    pub name: AtomToken,
    pub _open: SymbolToken,
    pub fields: Option<Sequence<RecordField<T>>>,
    pub _close: SymbolToken,
}
impl<T: Parse> Parse for Record<T> {
    fn parse<U: TokenRead>(parser: &mut Parser<U>) -> Result<Self> {
        Ok(Record {
            _sharp: track!(parser.expect(&Symbol::Sharp))?,
            name: track!(parser.parse())?,
            _open: track!(parser.expect(&Symbol::OpenBrace))?,
            fields: track!(parser.parse())?,
            _close: track!(parser.expect(&Symbol::CloseBrace))?,
        })
    }
}
impl<T> PositionRange for Record<T> {
    fn start_position(&self) -> Position {
        self._sharp.start_position()
    }
    fn end_position(&self) -> Position {
        self._close.end_position()
    }
}

/// `#` `AtomToken` `.` `AtomToken`
#[derive(Debug, Clone)]
pub struct RecordFieldIndex {
    pub _sharp: SymbolToken,
    pub name: AtomToken,
    pub _dot: SymbolToken,
    pub field: AtomToken,
}
impl Parse for RecordFieldIndex {
    fn parse<T: TokenRead>(parser: &mut Parser<T>) -> Result<Self> {
        Ok(RecordFieldIndex {
            _sharp: track!(parser.expect(&Symbol::Sharp))?,
            name: track!(parser.parse())?,
            _dot: track!(parser.expect(&Symbol::Dot))?,
            field: track!(parser.parse())?,
        })
    }
}
impl PositionRange for RecordFieldIndex {
    fn start_position(&self) -> Position {
        self._sharp.start_position()
    }
    fn end_position(&self) -> Position {
        self.field.end_position()
    }
}

/// `#` `{` `Option<Sequence<MapField<T>>>` `}`
#[derive(Debug, Clone)]
pub struct Map<T> {
    pub _sharp: SymbolToken,
    pub _open: SymbolToken,
    pub fields: Option<Sequence<MapField<T>>>,
    pub _close: SymbolToken,
}
impl<T: Parse> Parse for Map<T> {
    fn parse<U: TokenRead>(parser: &mut Parser<U>) -> Result<Self> {
        Ok(Map {
            _sharp: track!(parser.expect(&Symbol::Sharp))?,
            _open: track!(parser.expect(&Symbol::OpenBrace))?,
            fields: track!(parser.parse())?,
            _close: track!(parser.expect(&Symbol::CloseBrace))?,
        })
    }
}
impl<T> PositionRange for Map<T> {
    fn start_position(&self) -> Position {
        self._sharp.start_position()
    }
    fn end_position(&self) -> Position {
        self._close.end_position()
    }
}

/// `Pattern` `=` `T`
#[derive(Debug, Clone)]
pub struct Match<T> {
    pub pattern: Pattern,
    pub _match: SymbolToken,
    pub value: T,
}
impl<T: Parse> Parse for Match<T> {
    fn parse<U: TokenRead>(parser: &mut Parser<U>) -> Result<Self> {
        Ok(Match {
            pattern: track!(Pattern::parse_non_left_recor(parser))?,
            _match: track!(parser.expect(&Symbol::Match))?,
            value: track!(parser.parse())?,
        })
    }
}
impl<T: Parse> ParseTail for Match<T> {
    type Head = Pattern;
    fn parse_tail<U: TokenRead>(parser: &mut Parser<U>, head: Pattern) -> Result<Self> {
        Ok(Match {
            pattern: head,
            _match: track!(parser.expect(&Symbol::Match))?,
            value: track!(parser.parse())?,
        })
    }
}
impl<T: PositionRange> PositionRange for Match<T> {
    fn start_position(&self) -> Position {
        self.pattern.start_position()
    }
    fn end_position(&self) -> Position {
        self.value.end_position()
    }
}

/// `T` `BinaryOp` `T`
#[derive(Debug, Clone)]
pub struct BinaryOpCall<T> {
    pub left: T,
    pub op: BinaryOp,
    pub right: T,
}
impl<T: Parse> ParseTail for BinaryOpCall<T> {
    type Head = T;
    fn parse_tail<U: TokenRead>(parser: &mut Parser<U>, head: Self::Head) -> Result<Self> {
        Ok(BinaryOpCall {
            left: head,
            op: track!(parser.parse())?,
            right: track!(parser.parse())?,
        })
    }
}
impl<T: PositionRange> PositionRange for BinaryOpCall<T> {
    fn start_position(&self) -> Position {
        self.left.start_position()
    }
    fn end_position(&self) -> Position {
        self.right.end_position()
    }
}

/// `UnaryOp` `T`
#[derive(Debug, Clone)]
pub struct UnaryOpCall<T> {
    pub op: UnaryOp,
    pub operand: T,
}
impl<T: Parse> Parse for UnaryOpCall<T> {
    fn parse<U: TokenRead>(parser: &mut Parser<U>) -> Result<Self> {
        Ok(UnaryOpCall {
            op: track!(parser.parse())?,
            operand: track!(parser.parse())?,
        })
    }
}
impl<T: PositionRange> PositionRange for UnaryOpCall<T> {
    fn start_position(&self) -> Position {
        self.op.start_position()
    }
    fn end_position(&self) -> Position {
        self.operand.end_position()
    }
}

/// `Option<ModulePrefix<T>>` `T` `Args<A>`
#[derive(Debug, Clone)]
pub struct Call<T, A = T> {
    pub module: Option<ModulePrefix<T>>,
    pub name: T,
    pub args: Args<A>,
}
impl<T: Parse, A: Parse> Parse for Call<T, A> {
    fn parse<U: TokenRead>(parser: &mut Parser<U>) -> Result<Self> {
        Ok(Call {
            module: track!(parser.parse())?,
            name: track!(T::parse_non_left_recor(parser))?,
            args: track!(parser.parse())?,
        })
    }
}
impl<T: Parse, A: Parse> ParseTail for Call<T, A> {
    type Head = T;
    fn parse_tail<U: TokenRead>(parser: &mut Parser<U>, head: Self::Head) -> Result<Self> {
        if let Ok(_colon) = parser.transaction(|parser| parser.expect(&Symbol::Colon)) {
            Ok(Call {
                module: Some(ModulePrefix { name: head, _colon }),
                name: track!(T::parse_non_left_recor(parser))?,
                args: track!(parser.parse())?,
            })
        } else {
            Ok(Call {
                module: None,
                name: head,
                args: track!(parser.parse())?,
            })
        }
    }
}
impl<T: PositionRange, A> PositionRange for Call<T, A> {
    fn start_position(&self) -> Position {
        self.module
            .as_ref()
            .map(|x| x.start_position())
            .unwrap_or_else(|| self.name.start_position())
    }
    fn end_position(&self) -> Position {
        self.args.end_position()
    }
}

/// `(` `T` `)`
#[derive(Debug, Clone)]
pub struct Parenthesized<T> {
    pub _open: SymbolToken,
    pub item: T,
    pub _close: SymbolToken,
}
impl<T: Parse> Parse for Parenthesized<T> {
    fn parse<U: TokenRead>(parser: &mut Parser<U>) -> Result<Self> {
        Ok(Parenthesized {
            _open: track!(parser.expect(&Symbol::OpenParen))?,
            item: track!(parser.parse())?,
            _close: track!(parser.expect(&Symbol::CloseParen))?,
        })
    }
}
impl<T> PositionRange for Parenthesized<T> {
    fn start_position(&self) -> Position {
        self._open.start_position()
    }
    fn end_position(&self) -> Position {
        self._close.end_position()
    }
}

/// `AtomToken` | `VariableToken`
#[derive(Debug, Clone)]
pub enum AtomOrVariable {
    Atom(AtomToken),
    Variable(VariableToken),
}
impl AtomOrVariable {
    pub fn value(&self) -> &str {
        match *self {
            AtomOrVariable::Atom(ref t) => t.value(),
            AtomOrVariable::Variable(ref t) => t.value(),
        }
    }
    pub fn text(&self) -> &str {
        match *self {
            AtomOrVariable::Atom(ref t) => t.text(),
            AtomOrVariable::Variable(ref t) => t.text(),
        }
    }
}
impl Parse for AtomOrVariable {
    fn parse<T: TokenRead>(parser: &mut Parser<T>) -> Result<Self> {
        match track!(parser.parse())? {
            LexicalToken::Atom(token) => Ok(AtomOrVariable::Atom(token)),
            LexicalToken::Variable(token) => Ok(AtomOrVariable::Variable(token)),
            token => track_panic!(ErrorKind::UnexpectedToken(token)),
        }
    }
}
impl PositionRange for AtomOrVariable {
    fn start_position(&self) -> Position {
        match *self {
            AtomOrVariable::Atom(ref t) => t.start_position(),
            AtomOrVariable::Variable(ref t) => t.start_position(),
        }
    }
    fn end_position(&self) -> Position {
        match *self {
            AtomOrVariable::Atom(ref t) => t.end_position(),
            AtomOrVariable::Variable(ref t) => t.end_position(),
        }
    }
}

/// `IntegerToken` | `VariableToken`
#[derive(Debug, Clone)]
pub enum IntegerOrVariable {
    Integer(IntegerToken),
    Variable(VariableToken),
}
impl IntegerOrVariable {
    pub fn text(&self) -> &str {
        match *self {
            IntegerOrVariable::Integer(ref t) => t.text(),
            IntegerOrVariable::Variable(ref t) => t.text(),
        }
    }
}
impl Parse for IntegerOrVariable {
    fn parse<T: TokenRead>(parser: &mut Parser<T>) -> Result<Self> {
        match track!(parser.parse())? {
            LexicalToken::Integer(token) => Ok(IntegerOrVariable::Integer(token)),
            LexicalToken::Variable(token) => Ok(IntegerOrVariable::Variable(token)),
            token => track_panic!(ErrorKind::UnexpectedToken(token)),
        }
    }
}
impl PositionRange for IntegerOrVariable {
    fn start_position(&self) -> Position {
        match *self {
            IntegerOrVariable::Integer(ref t) => t.start_position(),
            IntegerOrVariable::Variable(ref t) => t.start_position(),
        }
    }
    fn end_position(&self) -> Position {
        match *self {
            IntegerOrVariable::Integer(ref t) => t.end_position(),
            IntegerOrVariable::Variable(ref t) => t.end_position(),
        }
    }
}
