use std::io::Write;

use firefly_rt::function::ErlangResult;
use firefly_rt::process::ProcessLock;
use firefly_rt::term::{OpaqueTerm, Term};

use crate::badarg;

#[export_name = "erlang:display/1"]
pub extern "C-unwind" fn display(_process: &mut ProcessLock, term: OpaqueTerm) -> ErlangResult {
    let term: Term = term.into();
    println!("{}", &term);
    ErlangResult::Ok(true.into())
}

#[export_name = "erlang:debug/1"]
pub extern "C-unwind" fn debug(_process: &mut ProcessLock, term: OpaqueTerm) -> ErlangResult {
    let term: Term = term.into();
    println!("{:?}", &term);
    ErlangResult::Ok(true.into())
}

#[export_name = "erlang:display_nl/0"]
pub extern "C-unwind" fn display_nl(_process: &mut ProcessLock) -> ErlangResult {
    println!();
    ErlangResult::Ok(true.into())
}

#[export_name = "erlang:display_string/1"]
pub extern "C-unwind" fn display_string(
    process: &mut ProcessLock,
    term: OpaqueTerm,
) -> ErlangResult {
    let list: Term = term.into();
    match list {
        Term::Nil => ErlangResult::Ok(true.into()),
        Term::Cons(cons) => {
            match cons.as_ref().to_string() {
                Some(ref s) => print!("{}", s),
                None => badarg!(process, term),
            }
            ErlangResult::Ok(true.into())
        }
        _other => badarg!(process, term),
    }
}

#[export_name = "erlang:puts/1"]
pub extern "C-unwind" fn puts(_process: &mut ProcessLock, printable: OpaqueTerm) -> ErlangResult {
    let printable: Term = printable.into();

    let bits = printable.as_bitstring().unwrap();
    assert!(bits.is_aligned());
    assert!(bits.is_binary());
    let bytes = unsafe { bits.as_bytes_unchecked() };
    let mut stdout = std::io::stdout().lock();
    stdout.write_all(bytes).unwrap();
    ErlangResult::Ok(true.into())
}
