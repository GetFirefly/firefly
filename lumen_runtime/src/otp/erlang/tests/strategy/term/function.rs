use proptest::strategy::Strategy;

use crate::term::Term;

pub fn module() -> impl Strategy<Value = Term> {
    super::atom()
}

pub fn function() -> impl Strategy<Value = Term> {
    super::atom()
}

pub fn arity() -> impl Strategy<Value = usize> {
    (0_usize..=255_usize).boxed()
}
