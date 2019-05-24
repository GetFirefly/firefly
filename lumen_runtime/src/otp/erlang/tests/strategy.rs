use std::sync::Arc;

use proptest::collection::SizeRange;
use proptest::strategy::{BoxedStrategy, Strategy};

use crate::process::Process;
use crate::term::Term;

pub mod term;

fn bits_to_bytes(bits: usize) -> usize {
    (bits + 7) / 8
}

fn byte_vec(size_range: SizeRange) -> BoxedStrategy<Vec<u8>> {
    proptest::collection::vec(proptest::prelude::any::<u8>(), size_range).boxed()
}

pub fn term(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    let max_len = 16;
    let size_range_inclusive = 0..=max_len;
    let size_range: SizeRange = size_range_inclusive.clone().into();

    let container_arc_process = arc_process.clone();

    term::leaf(size_range_inclusive, arc_process)
        .prop_recursive(4, 64, 16, move |element| {
            term::container(element, size_range.clone(), container_arc_process.clone())
        })
        .boxed()
}
