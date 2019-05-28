use std::sync::Arc;

use proptest::collection::SizeRange;
use proptest::strategy::Strategy;

use crate::otp::erlang::tests::strategy::{self, NON_EMPTY_RANGE_INCLUSIVE};
use crate::process::Process;
use crate::term::Term;

pub fn intermediate(
    element: impl Strategy<Value = Term>,
    size_range: SizeRange,
    arc_process: Arc<Process>,
) -> impl Strategy<Value = Term> {
    proptest::collection::vec(element, size_range).prop_map(move |vec| match vec.len() {
        0 => Term::EMPTY_LIST,
        1 => Term::slice_to_list(&vec, &arc_process),
        len => {
            let last_index = len - 1;

            Term::slice_to_improper_list(&vec[0..last_index], vec[last_index], &arc_process)
        }
    })
}

pub fn non_empty_maybe_improper(arc_process: Arc<Process>) -> impl Strategy<Value = Term> {
    let size_range: SizeRange = NON_EMPTY_RANGE_INCLUSIVE.clone().into();

    proptest::collection::vec(strategy::term(arc_process.clone()), size_range).prop_map(
        move |vec| match vec.len() {
            1 => Term::slice_to_list(&vec, &arc_process),
            len => {
                let last_index = len - 1;

                Term::slice_to_improper_list(&vec[0..last_index], vec[last_index], &arc_process)
            }
        },
    )
}
