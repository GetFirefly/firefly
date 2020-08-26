#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

#[native_implemented::function(erlang:append_element/2)]
fn result(process: &Process, tuple: Term, element: Term) -> exception::Result<Term> {
    let internal = term_try_into_tuple!(tuple)?;
    let mut new_elements_vec: Vec<Term> = Vec::with_capacity(internal.len() + 1);
    new_elements_vec.extend_from_slice(&internal[..]);
    new_elements_vec.push(element);
    let new_tuple = process.tuple_from_slice(&new_elements_vec);

    Ok(new_tuple)
}
