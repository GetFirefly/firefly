use trackable::track;

use crate::syntax::tokenizer::tokens::{IntegerToken, SymbolToken, VariableToken};
use crate::syntax::tokenizer::values::Symbol;
use crate::syntax::tokenizer::{Position, PositionRange};

use crate::syntax::parser::traits::{Parse, TokenRead};
use crate::syntax::parser::{Parser, Result};

use super::super::Type;

/// `Type` `Option<NonEmpty>`
#[derive(Debug, Clone)]
pub struct ListElement {
    pub element_type: Type,
    pub non_empty: Option<NonEmpty>,
}
impl Parse for ListElement {
    fn parse<T: TokenRead>(parser: &mut Parser<T>) -> Result<Self> {
        Ok(ListElement {
            element_type: track!(parser.parse())?,
            non_empty: track!(parser.parse())?,
        })
    }
}
impl PositionRange for ListElement {
    fn start_position(&self) -> Position {
        self.element_type.start_position()
    }
    fn end_position(&self) -> Position {
        self.non_empty
            .as_ref()
            .map(|t| t.end_position())
            .unwrap_or_else(|| self.element_type.end_position())
    }
}

/// `,` `...`
#[derive(Debug, Clone)]
pub struct NonEmpty {
    pub _comma: SymbolToken,
    pub _triple_dot: SymbolToken,
}
impl Parse for NonEmpty {
    fn parse<T: TokenRead>(parser: &mut Parser<T>) -> Result<Self> {
        Ok(NonEmpty {
            _comma: track!(parser.expect(&Symbol::Comma))?,
            _triple_dot: track!(parser.expect(&Symbol::TripleDot))?,
        })
    }
}
impl PositionRange for NonEmpty {
    fn start_position(&self) -> Position {
        self._comma.start_position()
    }
    fn end_position(&self) -> Position {
        self._triple_dot.end_position()
    }
}

/// `ByteSize` `,` `BitSize`
#[derive(Debug, Clone)]
pub struct ByteAndBitSize {
    pub byte: ByteSize,
    pub _comma: SymbolToken,
    pub bit: BitSize,
}
impl Parse for ByteAndBitSize {
    fn parse<T: TokenRead>(parser: &mut Parser<T>) -> Result<Self> {
        Ok(ByteAndBitSize {
            byte: track!(parser.parse())?,
            _comma: track!(parser.expect(&Symbol::Comma))?,
            bit: track!(parser.parse())?,
        })
    }
}
impl PositionRange for ByteAndBitSize {
    fn start_position(&self) -> Position {
        self.byte.start_position()
    }
    fn end_position(&self) -> Position {
        self.bit.end_position()
    }
}

/// `_` `:` `IntegerToken`
#[derive(Debug, Clone)]
pub struct ByteSize {
    pub _underscore: VariableToken,
    pub _colon: SymbolToken,
    pub size: IntegerToken,
}
impl Parse for ByteSize {
    fn parse<T: TokenRead>(parser: &mut Parser<T>) -> Result<Self> {
        Ok(ByteSize {
            _underscore: track!(parser.expect("_"))?,
            _colon: track!(parser.expect(&Symbol::Colon))?,
            size: track!(parser.parse())?,
        })
    }
}
impl PositionRange for ByteSize {
    fn start_position(&self) -> Position {
        self._underscore.start_position()
    }
    fn end_position(&self) -> Position {
        self.size.end_position()
    }
}

/// `_` `:` `_` `*` `IntegerToken`
#[derive(Debug, Clone)]
pub struct BitSize {
    pub _underscore0: VariableToken,
    pub _colon: SymbolToken,
    pub _underscore1: VariableToken,
    pub _asterisk: SymbolToken,
    pub size: IntegerToken,
}
impl Parse for BitSize {
    fn parse<T: TokenRead>(parser: &mut Parser<T>) -> Result<Self> {
        Ok(BitSize {
            _underscore0: track!(parser.expect("_"))?,
            _colon: track!(parser.expect(&Symbol::Colon))?,
            _underscore1: track!(parser.expect("_"))?,
            _asterisk: track!(parser.expect(&Symbol::Multiply))?,
            size: track!(parser.parse())?,
        })
    }
}
impl PositionRange for BitSize {
    fn start_position(&self) -> Position {
        self._underscore0.start_position()
    }
    fn end_position(&self) -> Position {
        self.size.end_position()
    }
}

/// `ByteAndBitSize` | `ByteSize` | `BitSize`
#[derive(Debug, Clone)]
#[cfg_attr(feature = "cargo-clippy", allow(large_enum_variant))]
pub enum BitsSpec {
    BytesAndBits(ByteAndBitSize),
    Bytes(ByteSize),
    Bits(BitSize),
}
impl Parse for BitsSpec {
    fn parse<T: TokenRead>(parser: &mut Parser<T>) -> Result<Self> {
        if let Ok(x) = parser.transaction(|parser| parser.parse()) {
            Ok(BitsSpec::BytesAndBits(x))
        } else if let Ok(x) = parser.transaction(|parser| parser.parse()) {
            Ok(BitsSpec::Bytes(x))
        } else {
            Ok(BitsSpec::Bits(track!(parser.parse())?))
        }
    }
}
impl PositionRange for BitsSpec {
    fn start_position(&self) -> Position {
        match *self {
            BitsSpec::BytesAndBits(ref t) => t.start_position(),
            BitsSpec::Bytes(ref t) => t.start_position(),
            BitsSpec::Bits(ref t) => t.start_position(),
        }
    }
    fn end_position(&self) -> Position {
        match *self {
            BitsSpec::BytesAndBits(ref t) => t.end_position(),
            BitsSpec::Bytes(ref t) => t.end_position(),
            BitsSpec::Bits(ref t) => t.end_position(),
        }
    }
}
