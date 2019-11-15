use std::sync::Arc;

use proptest::strategy::{BoxedStrategy, Strategy};

use liblumen_alloc::erts::term::prelude::Term;
use liblumen_alloc::erts::Process;

use crate::test::strategy::term::binary::sub::{bit_offset, byte_count, byte_offset};

pub fn is_not_empty(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    super::with_size_range(
        byte_offset(),
        bit_offset(),
        byte_count::non_empty(),
        (0_u8..=0_u8).boxed(),
        arc_process,
    )
}
