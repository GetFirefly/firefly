pub mod large;
pub mod small;

use liblumen_alloc::erts::exception::InternalResult;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::erts::Process;

use super::decode_vec_term;

fn decode<'a>(
    process: &Process,
    safe: bool,
    bytes: &'a [u8],
    len: usize,
) -> InternalResult<(Term, &'a [u8])> {
    let (element_vec, after_elements_vec) = decode_vec_term(process, safe, bytes, len)?;
    let tuple = process.tuple_from_slice(&element_vec)?;

    Ok((tuple, after_elements_vec))
}
