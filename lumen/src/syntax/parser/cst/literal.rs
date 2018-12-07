use trackable::{track, track_panic};

use crate::syntax::tokenizer::tokens::{
    AtomToken, CharToken, FloatToken, IntegerToken, StringToken,
};
use crate::syntax::tokenizer::{LexicalToken, Position, PositionRange};

use crate::syntax::parser::traits::{Parse, TokenRead};
use crate::syntax::parser::{ErrorKind, Parser, Result};

#[derive(Debug, Clone)]
pub enum Literal {
    Atom(AtomToken),
    Char(CharToken),
    Float(FloatToken),
    Integer(IntegerToken),
    String {
        head: StringToken,
        tail: Vec<StringToken>,
    },
}
impl Parse for Literal {
    fn parse<T>(parser: &mut Parser<T>) -> Result<Self>
    where
        T: TokenRead,
    {
        match track!(parser.parse())? {
            LexicalToken::Atom(t) => Ok(Literal::Atom(t)),
            LexicalToken::Char(t) => Ok(Literal::Char(t)),
            LexicalToken::Float(t) => Ok(Literal::Float(t)),
            LexicalToken::Integer(t) => Ok(Literal::Integer(t)),
            LexicalToken::String(head) => {
                let mut tail = Vec::new();
                while let Ok(t) = parser.transaction(|parser| parser.parse()) {
                    tail.push(t);
                }
                Ok(Literal::String { head, tail })
            }
            token => track_panic!(ErrorKind::UnexpectedToken(token)),
        }
    }
}
impl PositionRange for Literal {
    fn start_position(&self) -> Position {
        match *self {
            Literal::Atom(ref x) => x.start_position(),
            Literal::Char(ref x) => x.start_position(),
            Literal::Float(ref x) => x.start_position(),
            Literal::Integer(ref x) => x.start_position(),
            Literal::String { ref head, .. } => head.start_position(),
        }
    }
    fn end_position(&self) -> Position {
        match *self {
            Literal::Atom(ref x) => x.end_position(),
            Literal::Char(ref x) => x.end_position(),
            Literal::Float(ref x) => x.end_position(),
            Literal::Integer(ref x) => x.end_position(),
            Literal::String { ref head, ref tail } => tail
                .last()
                .map(|t| t.end_position())
                .unwrap_or_else(|| head.end_position()),
        }
    }
}
