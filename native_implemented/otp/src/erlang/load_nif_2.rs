use firefly_rt::process::Process;
use firefly_rt::term::{atoms, Term};

#[native_implemented::function(erlang:load_nif/2)]
pub fn result(process: &Process, _path: Term, _load_info: Term) -> Term {
    let reason: Term = atoms::Notsup.into();
    // Similar to the text used for HiPE compiled modules
    let text = process
        .list_from_chars("Calling load_nif from Lumen compiled modules not supported".chars());
    let tag: Term = atoms::Error.into();
    let value = process.tuple_term_from_term_slice(&[reason, text]);

    process.tuple_term_from_term_slice(&[tag, value])
}
