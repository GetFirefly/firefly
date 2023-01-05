#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::Term;

use crate::runtime::context::*;

#[native_implemented::function(erlang:make_tuple/2)]
pub fn result(
    process: &Process,
    arity: Term,
    initial_value: Term,
) -> Result<Term, NonNull<ErlangException>> {
    // arity by definition is only 0-225, so `u8`, but ...
    let arity_u8: u8 = term_try_into_arity(arity)?;
    // ... everything else uses `usize`, so cast it back up
    let arity_usize: usize = arity_u8 as usize;
    let element_vec: Vec<Term> = std::iter::repeat(initial_value).take(arity_usize).collect();

    Ok(process.tuple_term_from_term_slice(&element_vec))
}
