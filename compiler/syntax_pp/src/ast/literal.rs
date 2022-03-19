//! Atomic Literals
//!
//! See: [6.2 Atomic Literals](http://erlang.org/doc/apps/erts/absform.html#id87074)
use num::bigint::BigUint;
use num::traits::ToPrimitive;

use super::LineNum;

#[derive(Debug, Clone)]
pub struct Integer {
    pub line: LineNum,
    pub value: BigUint,
}
impl_node!(Integer);
impl Integer {
    pub fn new(line: LineNum, value: BigUint) -> Self {
        Integer { line, value }
    }
    pub fn to_u64(&self) -> Option<u64> {
        self.value.to_u64()
    }
}

#[derive(Debug, Clone)]
pub struct Char {
    pub line: LineNum,
    pub value: char,
}
impl_node!(Char);
impl Char {
    pub fn new(line: LineNum, value: char) -> Self {
        Char { line, value }
    }
}

#[derive(Debug, Clone)]
pub struct Float {
    pub line: LineNum,
    pub value: f64,
}
impl_node!(Float);
impl Float {
    pub fn new(line: LineNum, value: f64) -> Self {
        Float { line, value }
    }
}

#[derive(Debug, Clone)]
pub struct Str {
    pub line: LineNum,
    pub value: String,
}
impl_node!(Str);
impl Str {
    pub fn new(line: LineNum, value: String) -> Self {
        Str { line, value }
    }
}

#[derive(Debug, Clone)]
pub struct Atom {
    pub line: LineNum,
    pub value: String,
}
impl_node!(Atom);
impl Atom {
    pub fn new(line: LineNum, value: String) -> Self {
        Atom { line, value }
    }
}
