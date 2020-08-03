use std::sync::Arc;

use proptest::prop_oneof;
use proptest::strategy::{BoxedStrategy, Strategy};

use super::*;

pub fn isize() -> BoxedStrategy<isize> {
    prop_oneof![
        (std::isize::MIN..(SmallInteger::MIN_VALUE - 1)),
        ((SmallInteger::MAX_VALUE + 1)..std::isize::MAX)
    ]
    .boxed()
}

pub fn negative(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    (std::isize::MIN..(SmallInteger::MIN_VALUE - 1))
        .prop_map(move |i| arc_process.integer(i).unwrap())
        .boxed()
}

pub fn positive(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    ((SmallInteger::MAX_VALUE + 1)..std::isize::MAX)
        .prop_map(move |i| arc_process.integer(i).unwrap())
        .boxed()
}
