use std::sync::Arc;

use proptest::collection::SizeRange;
use proptest::strategy::Strategy;

use crate::process::Process;
use crate::term::Term;

pub fn intermediate(
    element: impl Strategy<Value = Term>,
    size_range: SizeRange,
    arc_process: Arc<Process>,
) -> impl Strategy<Value = Term> {
    proptest::collection::vec(element, size_range)
        .prop_map(move |vec| Term::slice_to_tuple(&vec, &arc_process))
}
