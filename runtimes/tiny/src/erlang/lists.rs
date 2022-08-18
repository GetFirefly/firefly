use liblumen_rt::backtrace::Trace;
use liblumen_rt::error::ErlangException;
use liblumen_rt::function::ErlangResult;
use liblumen_rt::term::*;

#[export_name = "lists:reverse/2"]
#[allow(improper_ctypes_definitions)]
pub extern "C-unwind" fn reverse(list: Term, tail: Term) -> ErlangResult {
    todo!()
}
