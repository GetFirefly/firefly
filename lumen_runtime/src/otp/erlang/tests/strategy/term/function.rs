use std::sync::Arc;

use proptest::prop_oneof;
use proptest::strategy::{BoxedStrategy, Strategy};

use crate::process::{IntoProcess, Process};
use crate::term::Term;

pub fn module() -> BoxedStrategy<Term> {
    super::atom()
}

pub fn function() -> BoxedStrategy<Term> {
    super::atom()
}

pub fn arity(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    arity_usize()
        .prop_map(move |u| u.into_process(&arc_process))
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
