use liblumen_rt::function::ErlangResult;
use liblumen_rt::term::*;

#[export_name = "erlang:display/1"]
pub extern "C-unwind" fn display(term: OpaqueTerm) -> ErlangResult {
    let term: Term = term.into();
    println!("{}", &term);
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
