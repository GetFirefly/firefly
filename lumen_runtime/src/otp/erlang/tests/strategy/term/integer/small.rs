use std::sync::Arc;

use proptest::strategy::{BoxedStrategy, Strategy};

use liblumen_alloc::erts::term::{SmallInteger, Term};
use liblumen_alloc::erts::ProcessControlBlock;

pub fn isize() -> BoxedStrategy<isize> {
    (SmallInteger::MIN_VALUE..=SmallInteger::MAX_VALUE).boxed()
}

pub fn negative(arc_process: Arc<ProcessControlBlock>) -> BoxedStrategy<Term> {
    (SmallInteger::MIN_VALUE..=-1)
        .prop_map(move |i| arc_process.integer(i))
        .boxed()
}

pub fn non_negative(arc_process: Arc<ProcessControlBlock>) -> BoxedStrategy<Term> {
    (0..=SmallInteger::MAX_VALUE)
        .prop_map(move |i| arc_process.integer(i))
        .boxed()
}

pub fn non_positive(arc_process: Arc<ProcessControlBlock>) -> BoxedStrategy<Term> {
    (SmallInteger::MIN_VALUE..=0)
        .prop_map(move |i| arc_process.integer(i))
        .boxed()
}

pub fn positive(arc_process: Arc<ProcessControlBlock>) -> BoxedStrategy<Term> {
    (1..=SmallInteger::MAX_VALUE)
        .prop_map(move |i| arc_process.integer(i))
        .boxed()
}
