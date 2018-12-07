use trackable::error::ErrorKindExt;
use trackable::track;

use crate::syntax::tokenizer::tokens::{AtomToken, SymbolToken, VariableToken};
use crate::syntax::tokenizer::values::Symbol;
use crate::syntax::tokenizer::{LexicalToken, Position, PositionRange};

use crate::syntax::parser::traits::{Parse, TokenRead};
use crate::syntax::parser::{ErrorKind, Parser, Result};

use super::commons::parts::{BinaryOp, UnaryOp};
use super::patterns;
use super::Literal;

#[derive(Debug, Clone)]
pub enum Pattern {
    Literal(Literal),
    Variable(VariableToken),
    Tuple(Box<patterns::Tuple>),
    Map(Box<patterns::Map>),
    Record(Box<patterns::Record>),
    RecordFieldIndex(Box<patterns::RecordFieldIndex>),
    List(Box<patterns::List>),
    Bits(Box<patterns::Bits>),
    Parenthesized(Box<patterns::Parenthesized>),
    UnaryOpCall(Box<patterns::UnaryOpCall>),
    BinaryOpCall(Box<patterns::BinaryOpCall>),
    Match(Box<patterns::Match>),
}
impl Parse for Pattern {
    fn parse_non_left_recor<T>(parser: &mut Parser<T>) -> Result<Self>
    where
        T: TokenRead,
    {
        let kind = track!(parser.peek(|parser| HeadKind::guess(parser)))?;
        let pattern = match kind {
            HeadKind::Literal => Pattern::Literal(track!(parser.parse())?),
            HeadKind::Variable => Pattern::Variable(track!(parser.parse())?),
            HeadKind::Tuple => Pattern::Tuple(track!(parser.parse())?),
            HeadKind::Map => Pattern::Map(track!(parser.parse())?),
            HeadKind::Record => Pattern::Record(track!(parser.parse())?),
            HeadKind::RecordFieldIndex => Pattern::RecordFieldIndex(track!(parser.parse())?),
            HeadKind::List => Pattern::List(track!(parser.parse())?),
            HeadKind::Bits => Pattern::Bits(track!(parser.parse())?),
            HeadKind::UnaryOpCall => Pattern::UnaryOpCall(track!(parser.parse())?),
            HeadKind::Parenthesized => Pattern::Parenthesized(track!(parser.parse())?),
        };
        Ok(pattern)
    }
    fn parse<T>(parser: &mut Parser<T>) -> Result<Self>
    where
        T: TokenRead,
    {
        let head = track!(Pattern::parse_non_left_recor(parser))?;
        let tail_kind = track!(parser.peek(|parser| TailKind::guess(parser)))?;
        match tail_kind {
            TailKind::BinaryOpCall => Ok(Pattern::BinaryOpCall(track!(parser.parse_tail(head))?)),
            TailKind::Match => Ok(Pattern::Match(track!(parser.parse_tail(head))?)),
            TailKind::None => Ok(head),
        }
    }
}
impl PositionRange for Pattern {
    fn start_position(&self) -> Position {
        match *self {
            Pattern::Literal(ref x) => x.start_position(),
            Pattern::Variable(ref x) => x.start_position(),
            Pattern::Tuple(ref x) => x.start_position(),
            Pattern::Map(ref x) => x.start_position(),
            Pattern::Record(ref x) => x.start_position(),
            Pattern::RecordFieldIndex(ref x) => x.start_position(),
            Pattern::List(ref x) => x.start_position(),
            Pattern::Bits(ref x) => x.start_position(),
            Pattern::Parenthesized(ref x) => x.start_position(),
            Pattern::UnaryOpCall(ref x) => x.start_position(),
            Pattern::BinaryOpCall(ref x) => x.start_position(),
            Pattern::Match(ref x) => x.start_position(),
        }
    }
    fn end_position(&self) -> Position {
        match *self {
            Pattern::Literal(ref x) => x.end_position(),
            Pattern::Variable(ref x) => x.end_position(),
            Pattern::Tuple(ref x) => x.end_position(),
            Pattern::Map(ref x) => x.end_position(),
            Pattern::Record(ref x) => x.end_position(),
            Pattern::RecordFieldIndex(ref x) => x.end_position(),
            Pattern::List(ref x) => x.end_position(),
            Pattern::Bits(ref x) => x.end_position(),
            Pattern::Parenthesized(ref x) => x.end_position(),
            Pattern::UnaryOpCall(ref x) => x.end_position(),
            Pattern::BinaryOpCall(ref x) => x.end_position(),
            Pattern::Match(ref x) => x.end_position(),
        }
    }
}

#[derive(Debug)]
enum HeadKind {
    Literal,
    Variable,
    Tuple,
    Map,
    Record,
    RecordFieldIndex,
    List,
    Bits,
    UnaryOpCall,
    Parenthesized,
}
impl HeadKind {
    fn guess<T: TokenRead>(parser: &mut Parser<T>) -> Result<Self> {
        Ok(match track!(parser.parse())? {
            LexicalToken::Symbol(t) => match t.value() {
                Symbol::OpenBrace => HeadKind::Tuple,
                Symbol::DoubleLeftAngle => HeadKind::Bits,
                Symbol::OpenParen => HeadKind::Parenthesized,
                Symbol::OpenSquare => HeadKind::List,
                Symbol::Sharp => {
                    if parser.parse::<AtomToken>().is_ok() {
                        let next = parser.parse::<SymbolToken>().map(|t| t.value()).ok();
                        if next == Some(Symbol::Dot) {
                            HeadKind::RecordFieldIndex
                        } else {
                            HeadKind::Record
                        }
                    } else {
                        HeadKind::Map
                    }
                }
                _ => track!(UnaryOp::from_token(t.into())
                    .map(|_| HeadKind::UnaryOpCall)
                    .map_err(|e| ErrorKind::UnexpectedToken(e).error()))?,
            },
            LexicalToken::Keyword(t) => track!(UnaryOp::from_token(t.into())
                .map(|_| HeadKind::UnaryOpCall)
                .map_err(|e| ErrorKind::UnexpectedToken(e).error()))?,
            LexicalToken::Variable(_) => HeadKind::Variable,
            _ => HeadKind::Literal,
        })
    }
}

#[derive(Debug)]
enum TailKind {
    BinaryOpCall,
    Match,
    None,
}
impl TailKind {
    fn guess<T: TokenRead>(parser: &mut Parser<T>) -> Result<Self> {
        let is_eos = track!(parser.eos())?;
        if is_eos {
            return Ok(TailKind::None);
        }
        Ok(match track!(parser.parse())? {
            LexicalToken::Symbol(ref t) if t.value() == Symbol::Match => TailKind::Match,
            token => {
                if BinaryOp::from_token(token).is_ok() {
                    TailKind::BinaryOpCall
                } else {
                    TailKind::None
                }
            }
        })
    }
}
