use firefly_rt::backtrace::Trace;
use firefly_rt::function::ErlangResult;
use firefly_rt::term::*;

use super::badarg;

#[export_name = "unicode:characters_to_list/2"]
#[allow(improper_ctypes_definitions)]
pub extern "C-unwind" fn characters_to_list(
    _data: OpaqueTerm,
    encoding: OpaqueTerm,
) -> ErlangResult {
    let Term::Atom(_encoding) = encoding.into() else { return badarg(Trace::capture()) };
    todo!()
}
