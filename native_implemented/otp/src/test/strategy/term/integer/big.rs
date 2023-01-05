use std::sync::Arc;

use proptest::prop_oneof;
use proptest::strategy::{BoxedStrategy, Strategy};
use firefly_rt::term::Integer;

use super::*;

pub fn isize() -> BoxedStrategy<isize> {
    prop_oneof![
        (std::isize::MIN..(Integer::MIN_SMALL - 1)),
        ((Integer::MAX_SMALL + 1)..std::isize::MAX)
    ]
    .boxed()
}

pub fn negative(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    (isize::MIN..(Integer::MIN_SMALL - 1))
        .prop_map(move |i| arc_process.integer(i).unwrap())
        .boxed()
}

pub fn positive(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    ((Integer::MAX_SMALL + 1)..isize::MAX)
        .prop_map(move |i| arc_process.integer(i).unwrap())
        .boxed()
}
