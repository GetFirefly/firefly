use std::ops::RangeInclusive;
use std::sync::Arc;

use proptest::collection::SizeRange;
use proptest::strategy::{BoxedStrategy, Strategy};

use crate::process::Process;
use crate::term::Term;

pub mod term;

pub fn term(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    let container_arc_process = arc_process.clone();

    term::leaf(SIZE_RANGE_INCLUSIVE, arc_process)
        .prop_recursive(4, 64, MAX_LEN as u32, move |element| {
            term::container(
                element,
                SIZE_RANGE_INCLUSIVE.clone().into(),
                container_arc_process.clone(),
            )
        })
        .boxed()
}

const MAX_LEN: usize = 16;
const SIZE_RANGE_INCLUSIVE: RangeInclusive<usize> = 0..=MAX_LEN;

fn bits_to_bytes(bits: usize) -> usize {
    (bits + 7) / 8
}

fn byte_vec(size_range: SizeRange) -> BoxedStrategy<Vec<u8>> {
    proptest::collection::vec(proptest::prelude::any::<u8>(), size_range).boxed()
}
