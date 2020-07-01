use std::sync::Arc;

use num_bigint::BigInt;

use proptest::strategy::{BoxedStrategy, Strategy};

use liblumen_alloc::erts::Process;

use crate::test::strategy;

use super::*;

pub fn is_one_based(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    (1_usize..std::usize::MAX)
        .prop_map(move |u| arc_process.integer(u).unwrap())
        .boxed()
}

pub fn is_not_one_based(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    strategy::term(arc_process)
        .prop_filter(
            "Index either must not be an integer or must be an integer <= 1",
            |index| match index.decode().unwrap() {
                TypedTerm::SmallInteger(small_integer) => {
                    let i: isize = small_integer.into();

                    i < 1
                }
                TypedTerm::BigInteger(big_integer) => {
                    let index_big_int: &BigInt = big_integer.as_ref().into();
                    let one_big_int: BigInt = 1.into();

                    index_big_int < &one_big_int
                }
                _ => true,
            },
        )
        .boxed()
}
