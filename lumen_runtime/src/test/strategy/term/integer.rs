use std::sync::Arc;

use proptest::prop_oneof;
use proptest::strategy::{BoxedStrategy, Strategy};

use liblumen_alloc::erts::term::{SmallInteger, Term};
use liblumen_alloc::erts::Process;

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
    (SmallInteger::MIN_VALUE..=SmallInteger::MAX_VALUE)
        .prop_map(move |i| arc_process.integer(i).unwrap())
        .boxed()
}
