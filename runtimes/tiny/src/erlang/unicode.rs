use liblumen_rt::backtrace::Trace;
use liblumen_rt::error::ErlangException;
use liblumen_rt::function::{self, ErlangResult, ModuleFunctionArity};
use liblumen_rt::term::*;

use crate::scheduler;

use super::badarg;

#[export_name = "unicode:characters_to_list/2"]
#[allow(improper_ctypes_definitions)]
pub extern "C-unwind" fn characters_to_list(
    data: OpaqueTerm,
    encoding: OpaqueTerm,
) -> ErlangResult {
    let Term::Atom(encoding) = encoding.into() else { return badarg(Trace::capture()) };
    todo!()
}
