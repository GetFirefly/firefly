use proptest::strategy::{BoxedStrategy, Strategy};

use crate::term::Term;

pub fn module() -> BoxedStrategy<Term> {
    super::atom()
}

pub fn function() -> BoxedStrategy<Term> {
    super::atom()
}

pub fn arity() -> BoxedStrategy<usize> {
    (0_usize..=255_usize).boxed()
}
