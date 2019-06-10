use std::sync::Arc;

use proptest::strategy::{BoxedStrategy, Strategy};

use crate::process::{IntoProcess, Process};
use crate::term::Term;

pub fn isize() -> BoxedStrategy<isize> {
    (crate::integer::small::MIN..=crate::integer::small::MAX).boxed()
}

pub fn negative(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    (crate::integer::small::MIN..=-1)
        .prop_map(move |i| i.into_process(&arc_process))
        .boxed()
}

pub fn non_negative(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    (0..=crate::integer::small::MAX)
        .prop_map(move |i| i.into_process(&arc_process))
        .boxed()
}

pub fn positive(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    (1..=crate::integer::small::MAX)
        .prop_map(move |i| i.into_process(&arc_process))
        .boxed()
}
