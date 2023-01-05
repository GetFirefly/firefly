use std::ptr::NonNull;

use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::Term;

#[native_implemented::function(erlang:append_element/2)]
fn result(process: &Process, tuple: Term, element: Term) -> Result<Term, NonNull<ErlangException>> {
    let internal = term_try_into_tuple!(tuple)?;
    let mut new_elements_vec: Vec<Term> = Vec::with_capacity(internal.len() + 1);
    new_elements_vec.extend_from_slice(&internal[..]);
    new_elements_vec.push(element);
    let new_tuple = process.tuple_term_from_term_slice(&new_elements_vec);

    Ok(new_tuple)
}
