#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::ptr::NonNull;

use anyhow::*;

use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::{Term, TypeError};

#[native_implemented::function(erlang:bit_size/1)]
pub fn result(process: &Process, bitstring: Term) -> Result<Term, NonNull<ErlangException>> {
    let option_total_bit_len = match bitstring {
        Term::ConstantBinary(binary_literal) => Some(binary_literal.total_bit_len()),
        Term::HeapBinary(heap_binary) => Some(heap_binary.total_bit_len()),
        Term::RcBinary(process_binary) => Some(process_binary.total_bit_len()),
        Term::RefBinary(subbinary) => Some(subbinary.total_bit_len()),
        _ => None,
    };

    match option_total_bit_len {
        Some(total_bit_len) => Ok(process.integer(total_bit_len).unwrap()),
        None => Err(TypeError)
            .context(format!("bitstring ({}) is not a bitstring", bitstring))
            .map_err(From::from),
    }
}
