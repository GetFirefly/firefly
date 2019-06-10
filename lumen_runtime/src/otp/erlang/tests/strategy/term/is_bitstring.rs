use std::ops::RangeInclusive;
use std::sync::Arc;

use proptest::prop_oneof;
use proptest::strategy::{BoxedStrategy, Strategy};

use crate::process::Process;
use crate::term::Term;

use super::binary::sub::{bit_count, bit_offset, byte_offset};
use super::binary::{heap, sub};

pub fn with_byte_len_range(
    byte_len_range_inclusive: RangeInclusive<usize>,
    arc_process: Arc<Process>,
) -> BoxedStrategy<Term> {
    prop_oneof![
        heap::with_size_range(byte_len_range_inclusive.clone().into(), arc_process.clone()),
        sub::with_size_range(
            byte_offset(),
            bit_offset(),
            byte_len_range_inclusive.boxed(),
            bit_count(),
            arc_process
        )
    ]
    .boxed()
}
