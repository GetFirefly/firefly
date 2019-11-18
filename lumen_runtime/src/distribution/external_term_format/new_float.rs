use liblumen_alloc::erts::exception::Exception;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::erts::Process;

use super::f64;

pub fn decode<'a>(process: &Process, bytes: &'a [u8]) -> Result<(Term, &'a [u8]), Exception> {
    let (f, after_f_bytes) = f64::decode(bytes)?;
    let float = process.float(f)?;

    Ok((float, after_f_bytes))
}
