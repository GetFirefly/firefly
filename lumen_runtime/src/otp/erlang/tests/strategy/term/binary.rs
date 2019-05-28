use std::sync::Arc;

use proptest::strategy::Strategy;

use crate::otp::erlang::tests::strategy::size_range;
use crate::otp::erlang::tests::strategy::term::binary::sub::{
    bit_count, bit_offset, byte_count, byte_offset,
};
use crate::process::Process;
use crate::term::Term;

pub mod heap;
pub mod sub;

pub fn heap(arc_process: Arc<Process>) -> impl Strategy<Value = Term> {
    heap::with_size_range(size_range(), arc_process)
}

pub fn sub(arc_process: Arc<Process>) -> impl Strategy<Value = Term> {
    sub::with_size_range(
        byte_offset(),
        bit_offset(),
        byte_count(),
        bit_count(),
        arc_process,
    )
}
