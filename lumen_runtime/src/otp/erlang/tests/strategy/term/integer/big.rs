use std::sync::Arc;

use proptest::prop_oneof;
use proptest::strategy::{BoxedStrategy, Strategy};

use liblumen_alloc::erts::term::{SmallInteger, Term};
use liblumen_alloc::erts::ProcessControlBlock;

pub fn isize() -> BoxedStrategy<isize> {
    prop_oneof![
        (std::isize::MIN..(SmallInteger::MIN_VALUE - 1)),
        ((SmallInteger::MAX_VALUE + 1)..std::isize::MAX)
    ]
    .boxed()
}

pub fn negative(arc_process: Arc<ProcessControlBlock>) -> BoxedStrategy<Term> {
    (std::isize::MIN..(SmallInteger::MIN_VALUE - 1))
        .prop_map(move |i| arc_process.integer(i))
        .boxed()
}

pub fn positive(arc_process: Arc<ProcessControlBlock>) -> BoxedStrategy<Term> {
    ((SmallInteger::MAX_VALUE + 1)..std::isize::MAX)
        .prop_map(move |i| arc_process.integer(i))
        .boxed()
}
