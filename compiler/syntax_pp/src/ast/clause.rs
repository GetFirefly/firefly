//! Clauses
//!
//! See: [6.5 Clauses](http://erlang.org/doc/apps/erts/absform.html#id88135)
use super::*;

#[derive(Debug, Clone)]
pub struct Clause {
    pub line: LineNum,
    pub patterns: Vec<Pattern>,
    pub guards: Vec<OrGuard>,
    pub body: Vec<Expression>,
}
impl Clause {
    pub fn new(
        line: LineNum,
        patterns: Vec<Pattern>,
        guards: Vec<OrGuard>,
        body: Vec<Expression>,
    ) -> Self {
        Clause {
            line,
            patterns,
            guards,
            body,
        }
    }
}
impl Node for Clause {
    fn line(&self) -> LineNum {
        self.line
    }
}
