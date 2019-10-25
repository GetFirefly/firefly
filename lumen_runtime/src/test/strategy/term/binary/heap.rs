use std::sync::Arc;

use proptest::collection::SizeRange;
use proptest::strategy::{BoxedStrategy, Strategy};

use liblumen_alloc::erts::term::prelude::Term;(
use liblumen_alloc::erts::Process;

use crate::test::strategy::byte_vec;

pub fn with_size_range(size_range: SizeRange, arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    byte_vec::with_size_range(size_range)
        .prop_map(move |byte_vec| arc_process.binary_from_bytes(&byte_vec).unwrap())
        .boxed()
}
