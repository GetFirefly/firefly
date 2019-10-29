use std::convert::TryInto;

use liblumen_alloc::badarg;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::erts::exception;

pub fn binary_to_string(binary: Term) -> exception::Result<String> {
    match binary.decode()? {
        TypedTerm::HeapBinary(heap_binary) => heap_binary.try_into(),
        TypedTerm::SubBinary(subbinary) => subbinary.try_into(),
        TypedTerm::ProcBin(process_binary) => process_binary.try_into(),
        TypedTerm::MatchContext(match_context) => match_context.try_into(),
        _ => Err(badarg!().into()),
    }
}
