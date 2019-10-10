use std::convert::TryInto;

use liblumen_alloc::badarg;
use liblumen_alloc::erts::exception::Exception;
use liblumen_alloc::erts::term::{Term, TypedTerm};

pub fn binary_to_string(binary: Term) -> Result<String, Exception> {
    match binary.to_typed_term().unwrap() {
        TypedTerm::Boxed(boxed) => match boxed.to_typed_term().unwrap() {
            TypedTerm::HeapBinary(heap_binary) => heap_binary.try_into(),
            TypedTerm::SubBinary(subbinary) => subbinary.try_into(),
            TypedTerm::ProcBin(process_binary) => process_binary.try_into(),
            TypedTerm::MatchContext(match_context) => match_context.try_into(),
            _ => Err(badarg!()),
        },
        _ => Err(badarg!()),
    }
    .map_err(|runtime_exception| runtime_exception.into())
}
