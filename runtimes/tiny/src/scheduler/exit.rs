use std::ptr::NonNull;

use liblumen_rt::error::{self, ErlangException};
use liblumen_rt::process::Process;
use liblumen_rt::term::{atoms, Term};

pub fn log_exit(process: &Process, ptr: NonNull<ErlangException>) -> bool {
    let exception = unsafe { ptr.as_ref() };
    let reason = exception.reason();

    if !is_expected_exit_reason(reason) {
        error::printer::print(process, exception).unwrap();
        true
    } else {
        false
    }
}

fn is_expected_exit_reason(reason: Term) -> bool {
    match reason {
        Term::Atom(a) if a == atoms::Normal => true,
        _ => false,
    }
}
