use trackable::{track, track_panic};

use crate::syntax::tokenizer::tokens::{AtomToken, SymbolToken, VariableToken};
use crate::syntax::tokenizer::values::{Keyword, Symbol};
use crate::syntax::tokenizer::{LexicalToken, Position, PositionRange};

use crate::syntax::parser::traits::{Parse, TokenRead};
use crate::syntax::parser::{ErrorKind, Parser, Result};

use super::commons::parts::BinaryOp;
use super::exprs;
use super::Literal;

#[derive(Debug, Clone)]
pub enum Expr {
    Literal(Literal),
    Variable(VariableToken),
    Tuple(Box<exprs::Tuple>),
    Map(Box<exprs::Map>),
    MapUpdate(Box<exprs::MapUpdate>),
    Record(Box<exprs::Record>),
    RecordUpdate(Box<exprs::RecordUpdate>),
    RecordFieldIndex(Box<exprs::RecordFieldIndex>),
    RecordFieldAccess(Box<exprs::RecordFieldAccess>),
    List(Box<exprs::List>),
    ListComprehension(Box<exprs::ListComprehension>),
    Bits(Box<exprs::Bits>),
    BitsComprehension(Box<exprs::BitsComprehension>),
    Fun(Box<exprs::Fun>),
    Parenthesized(Box<exprs::Parenthesized>),
    FunCall(Box<exprs::FunCall>),
    UnaryOpCall(Box<exprs::UnaryOpCall>),
    BinaryOpCall(Box<exprs::BinaryOpCall>),
    Match(Box<exprs::Match>),
    Block(Box<exprs::Block>),
    Catch(Box<exprs::Catch>),
    If(Box<exprs::If>),
    Case(Box<exprs::Case>),
    Receive(Box<exprs::Receive>),
    Try(Box<exprs::Try>),
}
impl Parse for Expr {
    fn parse_non_left_recor<T>(parser: &mut Parser<T>) -> Result<Self>
    where
        T: TokenRead,
    {
        let kind = track!(parser.peek(|parser| HeadKind::guess(parser)))?;
        let expr = match kind {
            HeadKind::Literal => Expr::Literal(track!(parser.parse())?),
            HeadKind::Variable => Expr::Variable(track!(parser.parse())?),
            HeadKind::Tuple => Expr::Tuple(track!(parser.parse())?),
            HeadKind::Map => Expr::Map(track!(parser.parse())?),
            HeadKind::Record => Expr::Record(track!(parser.parse())?),
            HeadKind::RecordFieldIndex => Expr::RecordFieldIndex(track!(parser.parse())?),
            HeadKind::List => Expr::List(track!(parser.parse())?),
            HeadKind::ListComprehension => Expr::ListComprehension(track!(parser.parse())?),
            HeadKind::Bits => Expr::Bits(track!(parser.parse())?),
            HeadKind::BitsComprehension => Expr::BitsComprehension(track!(parser.parse())?),
            HeadKind::Fun => Expr::Fun(track!(parser.parse())?),
            HeadKind::UnaryOpCall => Expr::UnaryOpCall(track!(parser.parse())?),
            HeadKind::Parenthesized => Expr::Parenthesized(track!(parser.parse())?),
            HeadKind::Block => Expr::Block(track!(parser.parse())?),
            HeadKind::Catch => Expr::Catch(track!(parser.parse())?),
            HeadKind::If => Expr::If(track!(parser.parse())?),
            HeadKind::Case => Expr::Case(track!(parser.parse())?),
            HeadKind::Receive => Expr::Receive(track!(parser.parse())?),
            HeadKind::Try => Expr::Try(track!(parser.parse())?),
            _ => track_panic!(ErrorKind::InvalidInput, "unreachable"),
        };
        Ok(expr)
    }
    fn parse<T>(parser: &mut Parser<T>) -> Result<Self>
    where
        T: TokenRead,
    {
        if let Ok(expr) = parser.transaction(|parser| parser.parse()) {
            return Ok(Expr::Match(expr));
        }

        let mut head = track!(Self::parse_non_left_recor(parser))?;
        loop {
            let kind = track!(parser.peek(|parser| TailKind::guess(parser)))?;
            head = match kind {
                TailKind::FunCall => Expr::FunCall(track!(parser.parse_tail(head))?),
                TailKind::MapUpdate => Expr::MapUpdate(track!(parser.parse_tail(head))?),
                TailKind::RecordUpdate => Expr::RecordUpdate(track!(parser.parse_tail(head))?),
                TailKind::RecordFieldAccess => {
                    Expr::RecordFieldAccess(track!(parser.parse_tail(head))?)
                }
                TailKind::BinaryOpCall => Expr::BinaryOpCall(track!(parser.parse_tail(head))?),
                TailKind::None => break,
            };
        }
        Ok(head)
    }
}
impl PositionRange for Expr {
    fn start_position(&self) -> Position {
        match *self {
            Expr::Literal(ref x) => x.start_position(),
            Expr::Variable(ref x) => x.start_position(),
            Expr::Tuple(ref x) => x.start_position(),
            Expr::Map(ref x) => x.start_position(),
            Expr::MapUpdate(ref x) => x.start_position(),
            Expr::Record(ref x) => x.start_position(),
            Expr::RecordUpdate(ref x) => x.start_position(),
            Expr::RecordFieldIndex(ref x) => x.start_position(),
            Expr::RecordFieldAccess(ref x) => x.start_position(),
            Expr::List(ref x) => x.start_position(),
            Expr::ListComprehension(ref x) => x.start_position(),
            Expr::Bits(ref x) => x.start_position(),
            Expr::BitsComprehension(ref x) => x.start_position(),
            Expr::Parenthesized(ref x) => x.start_position(),
            Expr::Fun(ref x) => x.start_position(),
            Expr::FunCall(ref x) => x.start_position(),
            Expr::UnaryOpCall(ref x) => x.start_position(),
            Expr::BinaryOpCall(ref x) => x.start_position(),
            Expr::Match(ref x) => x.start_position(),
            Expr::Block(ref x) => x.start_position(),
            Expr::Catch(ref x) => x.start_position(),
            Expr::If(ref x) => x.start_position(),
            Expr::Case(ref x) => x.start_position(),
            Expr::Receive(ref x) => x.start_position(),
            Expr::Try(ref x) => x.start_position(),
        }
    }
    fn end_position(&self) -> Position {
        match *self {
            Expr::Literal(ref x) => x.end_position(),
            Expr::Variable(ref x) => x.end_position(),
            Expr::Tuple(ref x) => x.end_position(),
            Expr::Map(ref x) => x.end_position(),
            Expr::MapUpdate(ref x) => x.end_position(),
            Expr::Record(ref x) => x.end_position(),
            Expr::RecordUpdate(ref x) => x.end_position(),
            Expr::RecordFieldIndex(ref x) => x.end_position(),
            Expr::RecordFieldAccess(ref x) => x.end_position(),
            Expr::List(ref x) => x.end_position(),
            Expr::ListComprehension(ref x) => x.end_position(),
            Expr::Bits(ref x) => x.end_position(),
            Expr::BitsComprehension(ref x) => x.end_position(),
            Expr::Parenthesized(ref x) => x.end_position(),
            Expr::Fun(ref x) => x.end_position(),
            Expr::FunCall(ref x) => x.end_position(),
            Expr::UnaryOpCall(ref x) => x.end_position(),
            Expr::BinaryOpCall(ref x) => x.end_position(),
            Expr::Match(ref x) => x.end_position(),
            Expr::Block(ref x) => x.end_position(),
            Expr::Catch(ref x) => x.end_position(),
            Expr::If(ref x) => x.end_position(),
            Expr::Case(ref x) => x.end_position(),
            Expr::Receive(ref x) => x.end_position(),
            Expr::Try(ref x) => x.end_position(),
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
    ListComprehension,
    Bits,
    BitsComprehension,
    Fun,
    UnaryOpCall,
    Parenthesized,
    Block,
    Catch,
    If,
    Case,
    Receive,
    Try,
    Annotated,
}
impl HeadKind {
    fn guess<T: TokenRead>(parser: &mut Parser<T>) -> Result<Self> {
        Ok(match track!(parser.parse())? {
            LexicalToken::Symbol(t) => match t.value() {
                Symbol::OpenBrace => HeadKind::Tuple,
                Symbol::DoubleLeftAngle => {
                    let maybe_comprehension = parser.parse::<Expr>().is_ok()
                        && parser
                            .expect::<SymbolToken>(&Symbol::DoubleVerticalBar)
                            .is_ok();
                    if maybe_comprehension {
                        HeadKind::BitsComprehension
                    } else {
                        HeadKind::Bits
                    }
                }
                Symbol::OpenParen => HeadKind::Parenthesized,
                Symbol::OpenSquare => {
                    let maybe_comprehension = parser.parse::<Expr>().is_ok()
                        && parser
                            .expect::<SymbolToken>(&Symbol::DoubleVerticalBar)
                            .is_ok();
                    if maybe_comprehension {
                        HeadKind::ListComprehension
                    } else {
                        HeadKind::List
                    }
                }
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
                Symbol::Plus | Symbol::Hyphen => HeadKind::UnaryOpCall,
                _ => track_panic!(ErrorKind::UnexpectedToken(t.into())),
            },
            LexicalToken::Keyword(t) => match t.value() {
                Keyword::Begin => HeadKind::Block,
                Keyword::Catch => HeadKind::Catch,
                Keyword::If => HeadKind::If,
                Keyword::Case => HeadKind::Case,
                Keyword::Receive => HeadKind::Receive,
                Keyword::Try => HeadKind::Try,
                Keyword::Fun => HeadKind::Fun,
                Keyword::Bnot | Keyword::Not => HeadKind::UnaryOpCall,
                _ => track_panic!(ErrorKind::UnexpectedToken(t.into())),
            },
            LexicalToken::Variable(_) => {
                if parser.expect::<SymbolToken>(&Symbol::DoubleColon).is_ok() {
                    HeadKind::Annotated
                } else {
                    HeadKind::Variable
                }
            }
            _ => HeadKind::Literal,
        })
    }
}

#[derive(Debug)]
enum TailKind {
    FunCall,
    MapUpdate,
    RecordUpdate,
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

        let token = track!(parser.parse::<LexicalToken>())?;
        Ok(match token.as_symbol_token().map(|t| t.value()) {
            Some(Symbol::OpenParen) | Some(Symbol::Colon) => TailKind::FunCall,
            Some(Symbol::Sharp) => {
                if parser
                    .parse::<LexicalToken>()
                    .ok()
                    .and_then(|t| t.as_atom_token().map(|_| ()))
                    .is_some()
                {
                    let is_record_update = parser
                        .parse::<LexicalToken>()
                        .ok()
                        .and_then(|t| t.as_symbol_token().map(|t| t.value() == Symbol::OpenBrace))
                        .unwrap_or(false);
                    if is_record_update {
                        TailKind::RecordUpdate
                    } else {
                        TailKind::RecordFieldAccess
                    }
                } else {
                    TailKind::MapUpdate
                }
            }
            _ => {
                if BinaryOp::from_token(token).is_ok() {
                    TailKind::BinaryOpCall
                } else {
                    TailKind::None
                }
            }
        })
    }
}
