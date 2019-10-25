use std::sync::Arc;

use num_bigint::BigInt;

use proptest::prop_oneof;
use proptest::strategy::{BoxedStrategy, Strategy};

use liblumen_alloc::erts::Process;

use super::*;

pub fn module() -> BoxedStrategy<Term> {
    super::atom()
}

pub fn function() -> BoxedStrategy<Term> {
    super::atom()
}

pub fn arity(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    arity_usize()
        .prop_map(move |u| arc_process.integer(u).unwrap())
        .boxed()
}

pub fn arity_or_arguments(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    prop_oneof![arity(arc_process.clone()), arguments(arc_process)].boxed()
}

pub fn arity_usize() -> BoxedStrategy<usize> {
    (0_usize..=255_usize).boxed()
}

pub fn arguments(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    super::list::proper(arc_process)
}

pub fn is_not_arity_or_arguments(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    super::super::term(arc_process)
        .prop_filter("Arity and argument must be neither an arity (>= 0) or arguments (an empty or non-empty proper list)", |term| match term.to_typed_term().unwrap() {
            TypedTerm::Nil => false,
            TypedTerm::List(cons) => !cons.is_proper(),
            TypedTerm::Boxed(boxed) => match boxed.to_typed_term().unwrap() {
                TypedTerm::BigInteger(big_integer) => {
                    let big_int: &BigInt = big_integer.as_ref().into();
                    let zero_big_int: &BigInt = &0.into();

                    big_int < zero_big_int
                }
                _ => true
            }
            TypedTerm::SmallInteger(small_integer) => {
                let i: isize = small_integer.into();

                i < 0
            }
            _ => true,
        })
        .boxed()
}
