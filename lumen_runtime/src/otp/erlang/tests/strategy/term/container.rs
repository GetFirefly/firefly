use super::{list, tuple};

use std::sync::Arc;

use proptest::collection::SizeRange;
use proptest::prop_oneof;
use proptest::strategy::{BoxedStrategy, Strategy};

use crate::process::Process;
use crate::term::Term;

// XXX work-around for bug related to sending maps across processes in `send_2` proptests
pub fn heap_fragment_safe(
    element: BoxedStrategy<Term>,
    size_range: SizeRange,
    arc_process: Arc<Process>,
) -> BoxedStrategy<Term> {
    prop_oneof![
        tuple::intermediate(element.clone(), size_range.clone(), arc_process.clone()),
        list::intermediate(element, size_range.clone(), arc_process.clone())
    ]
    .boxed()
}
