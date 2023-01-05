#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::ptr::NonNull;

use anyhow::*;

use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::Term;

#[native_implemented::function(erlang:size/1)]
pub fn result(process: &Process, binary_or_tuple: Term) -> Result<Term, NonNull<ErlangException>> {
    let option_size = match binary_or_tuple {
        Term::Tuple(tuple) => Some(tuple.len()),
        Term::HeapBinary(heap_binary) => Some(heap_binary.full_byte_len()),
        Term::RcBinary(process_binary) => Some(process_binary.full_byte_len()),
        Term::RefBinary(subbinary) => Some(subbinary.full_byte_len()),
        _ => None,
    };

    match option_size {
        Some(size) => Ok(process.integer(size).unwrap()),
        None => Err(TypeError)
            .context(format!(
                "binary_or_tuple ({}) is neither a binary nor a tuple",
                binary_or_tuple
            ))
            .map_err(From::from),
    }
}
