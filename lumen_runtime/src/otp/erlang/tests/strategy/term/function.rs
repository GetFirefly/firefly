use std::sync::Arc;

use proptest::prop_oneof;
use proptest::strategy::{BoxedStrategy, Strategy};

use liblumen_alloc::erts::term::Term;
use liblumen_alloc::erts::ProcessControlBlock;

pub fn module() -> BoxedStrategy<Term> {
    super::atom()
}

pub fn function() -> BoxedStrategy<Term> {
    super::atom()
}

pub fn arity(arc_process: Arc<ProcessControlBlock>) -> BoxedStrategy<Term> {
    arity_usize()
        .prop_map(move |u| arc_process.integer(u))
        .boxed()
}

pub fn arity_or_arguments(arc_process: Arc<ProcessControlBlock>) -> BoxedStrategy<Term> {
    prop_oneof![arity(arc_process.clone()), arguments(arc_process)].boxed()
}

pub fn arity_usize() -> BoxedStrategy<usize> {
    (0_usize..=255_usize).boxed()
}

pub fn arguments(arc_process: Arc<ProcessControlBlock>) -> BoxedStrategy<Term> {
    super::list::proper(arc_process)
}
