//! Clauses
//!
//! See: [6.5 Clauses](http://erlang.org/doc/apps/erts/absform.html#id88135)
use super::*;

#[derive(Debug, Clone)]
pub struct Clause {
    pub loc: Location,
    pub patterns: Vec<Expression>,
    pub guards: Vec<OrGuard>,
    pub body: Vec<Expression>,
}
impl_node!(Clause);
impl Clause {
    pub fn new(
        loc: Location,
        patterns: Vec<Expression>,
        guards: Vec<OrGuard>,
        body: Vec<Expression>,
    ) -> Self {
        Self {
            loc,
            patterns,
            guards,
            body,
        }
    }
}
