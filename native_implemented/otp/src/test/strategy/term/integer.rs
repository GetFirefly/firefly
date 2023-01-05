use std::sync::Arc;

use proptest::prop_oneof;
use proptest::strategy::{BoxedStrategy, Strategy};

use super::*;

pub mod big;
pub mod small;

pub fn big(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    prop_oneof![
        big::negative(arc_process.clone()),
        big::positive(arc_process)
    ]
    .boxed()
}

pub fn negative(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    prop_oneof![
        big::negative(arc_process.clone()),
        small::negative(arc_process)
    ]
    .boxed()
}

pub fn non_negative(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    prop_oneof![
        small::non_negative(arc_process.clone()),
        big::positive(arc_process)
    ]
    .boxed()
}

pub fn non_positive(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    prop_oneof![
        big::negative(arc_process.clone()),
        small::non_positive(arc_process)
    ]
    .boxed()
}

pub fn positive(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    prop_oneof![
        small::positive(arc_process.clone()),
        big::positive(arc_process)
    ]
    .boxed()
}

pub fn small(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    (Integer::MIN_SMALL..=Integer::MAX_SMALL)
        .prop_map(move |i| arc_process.integer(i).unwrap())
        .boxed()
}
