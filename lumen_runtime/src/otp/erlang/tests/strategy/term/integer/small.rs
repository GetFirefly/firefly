use std::sync::Arc;

use proptest::strategy::Strategy;

use crate::process::{IntoProcess, Process};
use crate::term::Term;

pub fn isize() -> impl Strategy<Value = isize> {
    (crate::integer::small::MIN..=crate::integer::small::MAX)
}

pub fn negative(arc_process: Arc<Process>) -> impl Strategy<Value = Term> {
    (crate::integer::small::MIN..=-1).prop_map(move |i| i.into_process(&arc_process))
}

pub fn non_negative(arc_process: Arc<Process>) -> impl Strategy<Value = Term> {
    (0..=crate::integer::small::MAX).prop_map(move |i| i.into_process(&arc_process))
}

pub fn positive(arc_process: Arc<Process>) -> impl Strategy<Value = Term> {
    (1..=crate::integer::small::MAX).prop_map(move |i| i.into_process(&arc_process))
}
