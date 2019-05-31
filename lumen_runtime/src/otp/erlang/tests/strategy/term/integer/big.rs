use std::sync::Arc;

use proptest::prop_oneof;
use proptest::strategy::{BoxedStrategy, Strategy};

use crate::process::{IntoProcess, Process};
use crate::term::Term;

pub fn isize() -> impl Strategy<Value = isize> {
    prop_oneof![
        (std::isize::MIN..(crate::integer::small::MIN - 1)),
        ((crate::integer::small::MAX + 1)..std::isize::MAX)
    ]
}

pub fn negative(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    (std::isize::MIN..(crate::integer::small::MIN - 1))
        .prop_map(move |i| i.into_process(&arc_process))
        .boxed()
}

pub fn positive(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    ((crate::integer::small::MAX + 1)..std::isize::MAX)
        .prop_map(move |i| i.into_process(&arc_process))
        .boxed()
}
