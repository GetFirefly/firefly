use std::sync::Arc;

use proptest::strategy::{BoxedStrategy, Strategy};

use crate::process::{IntoProcess, Process};
use crate::term::Term;

pub fn positive(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    (1..crate::integer::small::MAX)
        .prop_map(move |i| i.into_process(&arc_process))
        .boxed()
}
