use std::sync::Arc;

use proptest::collection::SizeRange;
use proptest::prop_oneof;
use proptest::strategy::{BoxedStrategy, Just, Strategy};

use crate::otp::erlang::tests::strategy::{self, NON_EMPTY_RANGE_INCLUSIVE};
use crate::process::Process;
use crate::term::Term;

pub fn improper(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    let size_range: SizeRange = NON_EMPTY_RANGE_INCLUSIVE.into();

    (
        proptest::collection::vec(strategy::term(arc_process.clone()), size_range),
        strategy::term::is_not_list(arc_process.clone()),
        Just(arc_process),
    )
        .prop_map(|(vec, tail, arc_process)| Term::slice_to_improper_list(&vec, tail, &arc_process))
        .boxed()
}

pub fn intermediate(
    element: BoxedStrategy<Term>,
    size_range: SizeRange,
    arc_process: Arc<Process>,
) -> BoxedStrategy<Term> {
    proptest::collection::vec(element, size_range)
        .prop_map(move |vec| match vec.len() {
            0 => Term::EMPTY_LIST,
            1 => Term::slice_to_list(&vec, &arc_process),
            len => {
                let last_index = len - 1;

                Term::slice_to_improper_list(&vec[0..last_index], vec[last_index], &arc_process)
            }
        })
        .boxed()
}

pub fn non_empty_maybe_improper(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    let size_range: SizeRange = NON_EMPTY_RANGE_INCLUSIVE.clone().into();

    proptest::collection::vec(strategy::term(arc_process.clone()), size_range)
        .prop_map(move |vec| match vec.len() {
            1 => Term::slice_to_list(&vec, &arc_process),
            len => {
                let last_index = len - 1;

                Term::slice_to_improper_list(&vec[0..last_index], vec[last_index], &arc_process)
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
        .prop_map(|(arc_process, vec)| Term::slice_to_list(&vec, &arc_process))
        .boxed()
}

pub fn proper(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    prop_oneof![Just(Term::EMPTY_LIST), non_empty_proper(arc_process)].boxed()
}
