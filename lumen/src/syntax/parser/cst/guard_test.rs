use trackable::error::ErrorKindExt;
use trackable::track;

use crate::syntax::tokenizer::tokens::{AtomToken, SymbolToken, VariableToken};
use crate::syntax::tokenizer::values::Symbol;
use crate::syntax::tokenizer::{LexicalToken, Position, PositionRange};

use crate::syntax::parser::traits::{Parse, TokenRead};
use crate::syntax::parser::{ErrorKind, Parser, Result};

use super::commons::parts::{BinaryOp, UnaryOp};
use super::guard_tests;
use super::Literal;

#[derive(Debug, Clone)]
pub enum GuardTest {
    Literal(Literal),
    Variable(VariableToken),
    Tuple(Box<guard_tests::Tuple>),
    Map(Box<guard_tests::Map>),
    Record(Box<guard_tests::Record>),
    RecordFieldIndex(Box<guard_tests::RecordFieldIndex>),
    RecordFieldAccess(Box<guard_tests::RecordFieldAccess>),
    List(Box<guard_tests::List>),
    Bits(Box<guard_tests::Bits>),
    Parenthesized(Box<guard_tests::Parenthesized>),
    FunCall(Box<guard_tests::FunCall>),
    UnaryOpCall(Box<guard_tests::UnaryOpCall>),
    BinaryOpCall(Box<guard_tests::BinaryOpCall>),
}
impl Parse for GuardTest {
    fn parse_non_left_recor<T>(parser: &mut Parser<T>) -> Result<Self>
    where
        T: TokenRead,
    {
        let kind = track!(parser.peek(|parser| HeadKind::guess(parser)))?;
        let test = match kind {
            HeadKind::Literal => GuardTest::Literal(track!(parser.parse())?),
            HeadKind::Variable => GuardTest::Variable(track!(parser.parse())?),
            HeadKind::Tuple => GuardTest::Tuple(track!(parser.parse())?),
            HeadKind::Map => GuardTest::Map(track!(parser.parse())?),
            HeadKind::Record => GuardTest::Record(track!(parser.parse())?),
            HeadKind::RecordFieldIndex => GuardTest::RecordFieldIndex(track!(parser.parse())?),
            HeadKind::List => GuardTest::List(track!(parser.parse())?),
            HeadKind::Bits => GuardTest::Bits(track!(parser.parse())?),
            HeadKind::FunCall => GuardTest::FunCall(track!(parser.parse())?),
            HeadKind::UnaryOpCall => GuardTest::UnaryOpCall(track!(parser.parse())?),
            HeadKind::Parenthesized => GuardTest::Parenthesized(track!(parser.parse())?),
        };
        Ok(test)
    }
    fn parse<T>(parser: &mut Parser<T>) -> Result<Self>
    where
        T: TokenRead,
    {
        let mut head = track!(Self::parse_non_left_recor(parser))?;
        loop {
            let kind = track!(parser.peek(|parser| TailKind::guess(parser)))?;
            head = match kind {
                TailKind::RecordFieldAccess => {
                    GuardTest::RecordFieldAccess(track!(parser.parse_tail(head))?)
                }
                TailKind::BinaryOpCall => GuardTest::BinaryOpCall(track!(parser.parse_tail(head))?),
                TailKind::None => break,
            };
        }
        Ok(head)
    }
}
impl PositionRange for GuardTest {
    fn start_position(&self) -> Position {
        match *self {
            GuardTest::Literal(ref x) => x.start_position(),
            GuardTest::Variable(ref x) => x.start_position(),
            GuardTest::Tuple(ref x) => x.start_position(),
            GuardTest::Map(ref x) => x.start_position(),
            GuardTest::Record(ref x) => x.start_position(),
            GuardTest::RecordFieldIndex(ref x) => x.start_position(),
            GuardTest::RecordFieldAccess(ref x) => x.start_position(),
            GuardTest::List(ref x) => x.start_position(),
            GuardTest::Bits(ref x) => x.start_position(),
            GuardTest::Parenthesized(ref x) => x.start_position(),
            GuardTest::FunCall(ref x) => x.start_position(),
            GuardTest::UnaryOpCall(ref x) => x.start_position(),
            GuardTest::BinaryOpCall(ref x) => x.start_position(),
        }
    }
    fn end_position(&self) -> Position {
        match *self {
            GuardTest::Literal(ref x) => x.end_position(),
            GuardTest::Variable(ref x) => x.end_position(),
            GuardTest::Tuple(ref x) => x.end_position(),
            GuardTest::Map(ref x) => x.end_position(),
            GuardTest::Record(ref x) => x.end_position(),
            GuardTest::RecordFieldIndex(ref x) => x.end_position(),
            GuardTest::RecordFieldAccess(ref x) => x.end_position(),
            GuardTest::List(ref x) => x.end_position(),
            GuardTest::Bits(ref x) => x.end_position(),
            GuardTest::Parenthesized(ref x) => x.end_position(),
            GuardTest::FunCall(ref x) => x.end_position(),
            GuardTest::UnaryOpCall(ref x) => x.end_position(),
            GuardTest::BinaryOpCall(ref x) => x.end_position(),
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
    FunCall,
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
            LexicalToken::Atom(_) => {
                let token = parser.parse::<SymbolToken>();
                match token.ok().map(|t| t.value()) {
                    Some(Symbol::OpenParen) | Some(Symbol::Colon) => HeadKind::FunCall,
                    _ => HeadKind::Literal,
                }
            }
            _ => HeadKind::Literal,
        })
    }
}

#[derive(Debug)]
enum TailKind {
    RecordFieldAccess,
    BinaryOpCall,
    None,
}
impl TailKind {
    fn guess<T: TokenRead>(parser: &mut Parser<T>) -> Result<Self> {
        let is_eos = track!(parser.eos())?;
        if is_eos {
            return Ok(TailKind::None);
        }
        Ok(match track!(parser.parse())? {
            LexicalToken::Symbol(ref t) if t.value() == Symbol::Sharp => {
                TailKind::RecordFieldAccess
            }
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
