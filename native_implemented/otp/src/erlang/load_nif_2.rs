use liblumen_alloc::atom;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::Term;

#[native_implemented::function(erlang:load_nif/2)]
pub fn result(process: &Process, _path: Term, _load_info: Term) -> Term {
    let reason = atom!("notsup");
    // Similar to the text used for HiPE compiled modules
    let text = process
        .list_from_chars("Calling load_nif from Lumen compiled modules not supported".chars());
    let tag = atom!("error");
    let value = process.tuple_from_slice(&[reason, text]);

    process.tuple_from_slice(&[tag, value])
}
