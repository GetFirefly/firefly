use std::sync::Arc;

use proptest::collection::SizeRange;
use proptest::strategy::{BoxedStrategy, Strategy};

use firefly_rt::process::Process;
use firefly_rt::term::Term;

use crate::test::strategy::byte_vec;

pub fn with_size_range(size_range: SizeRange, arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    byte_vec::with_size_range(size_range)
        .prop_map(move |byte_vec| arc_process.binary_from_bytes(&byte_vec))
        .boxed()
}
