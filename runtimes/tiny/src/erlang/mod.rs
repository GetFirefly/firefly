use std::io::Write;

use liblumen_rt::function::ErlangResult;
use liblumen_rt::term::*;

#[export_name = "erlang:display/1"]
pub extern "C-unwind" fn display(term: OpaqueTerm) -> ErlangResult {
    let term: Term = term.into();
    println!("{}", &term);
    Ok(true.into())
}

#[export_name = "erlang:puts/1"]
pub extern "C-unwind" fn puts(printable: OpaqueTerm) -> ErlangResult {
    let printable: Term = printable.into();

    let bits = printable.as_bitstring().unwrap();
    assert!(bits.is_aligned());
    assert!(bits.is_binary());
    let bytes = unsafe { bits.as_bytes_unchecked() };
    let mut stdout = std::io::stdout().lock();
    stdout.write_all(bytes).unwrap();
    Ok(true.into())
}

#[export_name = "erlang:is_atom/1"]
pub extern "C-unwind" fn is_atom(term: OpaqueTerm) -> ErlangResult {
    Ok(term.is_atom().into())
}

#[export_name = "erlang:=:=/2"]
pub extern "C-unwind" fn exact_eq(lhs: OpaqueTerm, rhs: OpaqueTerm) -> ErlangResult {
    let lhs: Term = lhs.into();
    let rhs: Term = rhs.into();
    Ok(lhs.exact_eq(&rhs).into())
}
