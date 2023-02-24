//! Atomic Literals
//!
//! See: [6.2 Atomic Literals](http://erlang.org/doc/apps/erts/absform.html#id87074)
use firefly_intern::Symbol;
use firefly_number::{self as number, ToPrimitive};

use super::Location;

#[derive(Debug, Clone)]
pub struct Integer {
    pub loc: Location,
    pub value: number::Int,
}
impl_node!(Integer);
impl Integer {
    pub fn new(loc: Location, value: number::Int) -> Self {
        Self { loc, value }
    }

    pub fn to_u64(&self) -> Option<u64> {
        self.value.to_u64()
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Char {
    pub loc: Location,
    pub value: char,
}
impl_node!(Char);
impl Char {
    pub fn new(loc: Location, value: char) -> Self {
        Self { loc, value }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Float {
    pub loc: Location,
    pub value: number::Float,
}
impl_node!(Float);
impl Float {
    pub fn new(loc: Location, value: number::Float) -> Self {
        Self { loc, value }
    }
}

#[derive(Debug, Clone)]
pub struct Str {
    pub loc: Location,
    pub value: Symbol,
}
impl_node!(Str);
impl Str {
    pub fn new(loc: Location, value: Symbol) -> Self {
        Self { loc, value }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Atom {
    pub loc: Location,
    pub value: Symbol,
}
impl_node!(Atom);
impl Atom {
    pub fn new(loc: Location, value: Symbol) -> Self {
        Self { loc, value }
    }
}
