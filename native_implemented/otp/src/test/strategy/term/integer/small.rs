use std::sync::Arc;

use proptest::strategy::{BoxedStrategy, Strategy};

use firefly_rt::process::Process;
use firefly_rt::term::{Integer, Term};

pub fn isize() -> BoxedStrategy<isize> {
    (Integer::MIN_SMALL..=Integer::MAX_SMALL).boxed()
}

pub fn negative(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    (Integer::MIN_SMALL..=-1)
        .prop_map(move |i| arc_process.integer(i).unwrap())
        .boxed()
}

pub fn non_negative(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    (0..=Integer::MAX_SMALL)
        .prop_map(move |i| arc_process.integer(i).unwrap())
        .boxed()
}

pub fn non_positive(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    (Integer::MIN_SMALL..=0)
        .prop_map(move |i| arc_process.integer(i).unwrap())
        .boxed()
}

pub fn positive(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    (1..=Integer::MAX_SMALL)
        .prop_map(move |i| arc_process.integer(i).unwrap())
        .boxed()
}
