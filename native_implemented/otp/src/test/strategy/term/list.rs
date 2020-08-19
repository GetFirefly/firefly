pub mod io;

use std::sync::Arc;

use proptest::collection::SizeRange;
use proptest::prop_oneof;
use proptest::strategy::{BoxedStrategy, Just, Strategy};

use liblumen_alloc::erts::term::prelude::Term;
use liblumen_alloc::erts::Process;

use crate::test::strategy::{self, NON_EMPTY_RANGE_INCLUSIVE};

pub fn improper(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    let size_range: SizeRange = NON_EMPTY_RANGE_INCLUSIVE.into();

    (
        proptest::collection::vec(strategy::term(arc_process.clone()), size_range),
        strategy::term::is_not_list(arc_process.clone()),
        Just(arc_process.clone()),
    )
        .prop_map(|(vec, tail, arc_process)| arc_process.improper_list_from_slice(&vec, tail))
        .boxed()
}

pub fn intermediate(
    element: BoxedStrategy<Term>,
    size_range: SizeRange,
    arc_process: Arc<Process>,
) -> BoxedStrategy<Term> {
    proptest::collection::vec(element, size_range)
        .prop_map(move |vec| match vec.len() {
            0 => Term::NIL,
            1 => arc_process.list_from_slice(&vec),
            len => {
                let last_index = len - 1;

                arc_process.improper_list_from_slice(&vec[0..last_index], vec[last_index])
            }
        })
        .boxed()
}

pub fn non_empty_maybe_improper(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    let size_range: SizeRange = NON_EMPTY_RANGE_INCLUSIVE.clone().into();

    proptest::collection::vec(strategy::term(arc_process.clone()), size_range)
        .prop_map(move |vec| match vec.len() {
            1 => arc_process.list_from_slice(&vec),
            len => {
                let last_index = len - 1;

                arc_process.improper_list_from_slice(&vec[0..last_index], vec[last_index])
            }
        })
        .boxed()
}

pub fn non_empty_proper(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    let size_range: SizeRange = NON_EMPTY_RANGE_INCLUSIVE.clone().into();

    (
        Just(arc_process.clone()),
        proptest::collection::vec(strategy::term(arc_process), size_range),
    )
        .prop_map(|(arc_process, vec)| arc_process.list_from_slice(&vec))
        .boxed()
}

pub fn proper(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    prop_oneof![Just(Term::NIL), non_empty_proper(arc_process)].boxed()
}
