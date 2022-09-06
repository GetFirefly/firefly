//! Guards
//!
//! See: [6.6 Guards](http://erlang.org/doc/apps/erts/absform.html#id88356)
use super::*;

#[derive(Debug, Clone)]
pub struct OrGuard {
    pub and_guards: Vec<Expression>,
}
impl OrGuard {
    pub fn new(and_guards: Vec<Expression>) -> Self {
        Self { and_guards }
    }
}
