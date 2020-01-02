use std::convert::TryInto;

use anyhow::*;

use liblumen_alloc::erts::exception::{self, Exception};
use liblumen_alloc::erts::term::prelude::*;

pub fn binary_to_string(binary: Term) -> exception::Result<String> {
    match binary.decode()? {
        TypedTerm::HeapBinary(heap_binary) => heap_binary.try_into().map_err(Exception::from),
        TypedTerm::SubBinary(subbinary) => subbinary.try_into().map_err(Exception::from),
        TypedTerm::ProcBin(process_binary) => process_binary.try_into().map_err(Exception::from),
        TypedTerm::MatchContext(match_context) => match_context.try_into().map_err(Exception::from),
        TypedTerm::BinaryLiteral(binary_literal) => {
            binary_literal.try_into().map_err(Exception::from)
        }
        _ => Err(TypeError)
            .context(format!("binary ({}) must be a binary", binary))
            .map_err(Exception::from),
    }
}
