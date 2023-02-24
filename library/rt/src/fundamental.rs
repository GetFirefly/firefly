///! The functions defined in this module are parts of the standard library that implement
///! fundamental functions of the Erlang language, e.g. operators, guards, etc. These are
///! not sensitive to differences in runtime, and so are implemented here to be shared by
///! all runtime implementations.
///!
///! Many of these are implemented on top of compiler intrinisics
use crate::function::ErlangResult;
use crate::intrinsics;
use crate::process::ProcessLock;
use crate::term::OpaqueTerm;

#[export_name = "erlang:is_atom/1"]
pub extern "C" fn is_atom1(_process: &mut ProcessLock, value: OpaqueTerm) -> ErlangResult {
    Ok(intrinsics::is_atom(value).into()).into()
}

#[export_name = "erlang:is_list/1"]
pub extern "C" fn is_list1(_process: &mut ProcessLock, value: OpaqueTerm) -> ErlangResult {
    Ok(intrinsics::is_list(value).into()).into()
}

#[export_name = "erlang:is_binary/1"]
pub extern "C" fn is_binary1(_process: &mut ProcessLock, value: OpaqueTerm) -> ErlangResult {
    Ok(intrinsics::is_binary(value).into()).into()
}

#[export_name = "erlang:is_function/1"]
pub extern "C" fn is_function1(_process: &mut ProcessLock, value: OpaqueTerm) -> ErlangResult {
    Ok(intrinsics::is_function(value).into()).into()
}

#[export_name = "erlang:is_big_integer/1"]
pub extern "C" fn is_big_integer1(_process: &mut ProcessLock, value: OpaqueTerm) -> ErlangResult {
    Ok(intrinsics::is_big_integer(value).into()).into()
}

#[export_name = "erlang:>=/2"]
pub extern "C" fn gte2(
    _process: &mut ProcessLock,
    lhs: OpaqueTerm,
    rhs: OpaqueTerm,
) -> ErlangResult {
    Ok(intrinsics::gte(lhs, rhs).into()).into()
}

#[export_name = "erlang:>/2"]
pub extern "C" fn gt2(
    _process: &mut ProcessLock,
    lhs: OpaqueTerm,
    rhs: OpaqueTerm,
) -> ErlangResult {
    Ok(intrinsics::gt(lhs, rhs).into()).into()
}

#[export_name = "erlang:</2"]
pub extern "C" fn lt2(
    _process: &mut ProcessLock,
    lhs: OpaqueTerm,
    rhs: OpaqueTerm,
) -> ErlangResult {
    Ok(intrinsics::lt(lhs, rhs).into()).into()
}

#[export_name = "erlang:=</2"]
pub extern "C" fn lte2(
    _process: &mut ProcessLock,
    lhs: OpaqueTerm,
    rhs: OpaqueTerm,
) -> ErlangResult {
    Ok(intrinsics::lte(lhs, rhs).into()).into()
}

#[export_name = "erlang:==/2"]
pub extern "C" fn eq2(
    _process: &mut ProcessLock,
    lhs: OpaqueTerm,
    rhs: OpaqueTerm,
) -> ErlangResult {
    Ok(intrinsics::eq(lhs, rhs).into()).into()
}

#[export_name = "erlang:/=/2"]
pub extern "C" fn ne2(
    _process: &mut ProcessLock,
    lhs: OpaqueTerm,
    rhs: OpaqueTerm,
) -> ErlangResult {
    Ok(intrinsics::ne(lhs, rhs).into()).into()
}

#[export_name = "erlang:=:=/2"]
pub extern "C" fn exact_eq2(
    _process: &mut ProcessLock,
    lhs: OpaqueTerm,
    rhs: OpaqueTerm,
) -> ErlangResult {
    Ok(intrinsics::exact_eq(lhs, rhs).into()).into()
}

#[export_name = "erlang:=/=/2"]
pub extern "C" fn exact_ne2(
    _process: &mut ProcessLock,
    lhs: OpaqueTerm,
    rhs: OpaqueTerm,
) -> ErlangResult {
    Ok(intrinsics::exact_ne(lhs, rhs).into()).into()
}
