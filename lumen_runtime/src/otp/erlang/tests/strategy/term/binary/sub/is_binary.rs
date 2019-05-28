use std::sync::Arc;

use proptest::strategy::Strategy;

use crate::otp::erlang::tests::strategy::term::binary::sub::{bit_offset, byte_count, byte_offset};
use crate::process::Process;
use crate::term::Term;

pub fn is_not_empty(arc_process: Arc<Process>) -> impl Strategy<Value = Term> {
    super::with_size_range(
        byte_offset(),
        bit_offset(),
        byte_count::non_empty(),
        0_u8..=0_u8,
        arc_process,
    )
}
