use liblumen_alloc::erts::exception::InternalResult;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::erts::Process;

use super::f64;

pub fn decode<'a>(process: &Process, bytes: &'a [u8]) -> InternalResult<(Term, &'a [u8])> {
    let (f, after_f_bytes) = f64::decode(bytes)?;
    let float = process.float(f)?;

    Ok((float, after_f_bytes))
}
