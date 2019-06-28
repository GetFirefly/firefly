use std::sync::Arc;

use proptest::collection::SizeRange;
use proptest::strategy::{BoxedStrategy, Strategy};

use crate::otp::erlang::tests::strategy::byte_vec;
use crate::process::Process;
use crate::term::Term;

pub fn with_size_range(size_range: SizeRange, arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    byte_vec::with_size_range(size_range)
        .prop_map(move |byte_vec| Term::slice_to_binary(&byte_vec, &arc_process))
        .boxed()
}
