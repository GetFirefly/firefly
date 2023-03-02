///! The functions defined in this module are parts of the standard library that implement
///! fundamental functions of the Erlang language, e.g. operators, guards, etc. These are
///! not sensitive to differences in runtime, and so are implemented here to be shared by
///! all runtime implementations.
///!
///! Many of these are implemented on top of compiler intrinisics
use crate::error::ExceptionInfo;
use crate::function::ErlangResult;
use crate::intrinsics;
use crate::process::ProcessLock;
use crate::term::{atoms, OpaqueTerm, Term, TermType};

#[export_name = "erlang:error/1"]
pub extern "C" fn error1(process: &mut ProcessLock, reason: OpaqueTerm) -> ErlangResult {
    process.exception_info = ExceptionInfo::error(reason);
    return ErlangResult::Err;
}

#[export_name = "erlang:error/2"]
pub extern "C" fn error2(
    process: &mut ProcessLock,
    reason: OpaqueTerm,
    args: OpaqueTerm,
) -> ErlangResult {
    process.exception_info = ExceptionInfo::error(reason);
    process.exception_info.args = Some(args);
    return ErlangResult::Err;
}

#[export_name = "erlang:error/3"]
pub extern "C" fn error3(
    process: &mut ProcessLock,
    reason: OpaqueTerm,
    args: OpaqueTerm,
    opts: OpaqueTerm,
) -> ErlangResult {
    let opts: Term = opts.into();
    let cause = opts.try_into().ok();
    process.exception_info = ExceptionInfo::error(reason);
    process.exception_info.args = Some(args);
    process.exception_info.cause = cause;
    return ErlangResult::Err;
}

#[export_name = "erlang:nif_error/1"]
pub extern "C" fn nif_error1(process: &mut ProcessLock, reason: OpaqueTerm) -> ErlangResult {
    error1(process, reason)
}

#[export_name = "erlang:is_atom/1"]
pub extern "C" fn is_atom1(_process: &mut ProcessLock, value: OpaqueTerm) -> ErlangResult {
    Ok(intrinsics::is_atom(value).into()).into()
}

#[export_name = "erlang:is_boolean/1"]
pub extern "C" fn is_boolean1(_process: &mut ProcessLock, value: OpaqueTerm) -> ErlangResult {
    match value {
        OpaqueTerm::TRUE | OpaqueTerm::FALSE => ErlangResult::Ok(OpaqueTerm::TRUE),
        _ => ErlangResult::Ok(OpaqueTerm::FALSE),
    }
}

#[export_name = "erlang:is_float/1"]
pub extern "C" fn is_float(_process: &mut ProcessLock, value: OpaqueTerm) -> ErlangResult {
    ErlangResult::Ok(value.is_float().into())
}

#[export_name = "erlang:is_integer/1"]
pub extern "C" fn is_integer1(_process: &mut ProcessLock, value: OpaqueTerm) -> ErlangResult {
    ErlangResult::Ok((value.r#typeof() == TermType::Int).into())
}

#[export_name = "erlang:is_number/1"]
pub extern "C" fn is_number1(_process: &mut ProcessLock, value: OpaqueTerm) -> ErlangResult {
    ErlangResult::Ok(value.r#typeof().is_number().into())
}

#[export_name = "erlang:is_list/1"]
pub extern "C" fn is_list1(_process: &mut ProcessLock, value: OpaqueTerm) -> ErlangResult {
    Ok(intrinsics::is_list(value).into()).into()
}

#[export_name = "erlang:is_map/1"]
pub extern "C" fn is_map1(_process: &mut ProcessLock, value: OpaqueTerm) -> ErlangResult {
    ErlangResult::Ok((value.r#typeof() == TermType::Map).into())
}

#[export_name = "erlang:is_pid/1"]
pub extern "C" fn is_pid1(_process: &mut ProcessLock, value: OpaqueTerm) -> ErlangResult {
    ErlangResult::Ok((value.r#typeof() == TermType::Pid).into())
}

#[export_name = "erlang:is_port/1"]
pub extern "C" fn is_port1(_process: &mut ProcessLock, value: OpaqueTerm) -> ErlangResult {
    ErlangResult::Ok((value.r#typeof() == TermType::Port).into())
}

#[export_name = "erlang:is_binary/1"]
pub extern "C" fn is_binary1(_process: &mut ProcessLock, value: OpaqueTerm) -> ErlangResult {
    Ok(intrinsics::is_binary(value).into()).into()
}

#[export_name = "erlang:is_bitstring/1"]
pub extern "C" fn is_bitstring1(_process: &mut ProcessLock, value: OpaqueTerm) -> ErlangResult {
    let term: Term = value.into();
    ErlangResult::Ok(term.is_bitstring().into())
}

#[export_name = "erlang:is_function/1"]
pub extern "C" fn is_function1(_process: &mut ProcessLock, value: OpaqueTerm) -> ErlangResult {
    Ok(intrinsics::is_function(value).into()).into()
}

#[export_name = "erlang:is_function/2"]
pub extern "C" fn is_function2(
    process: &mut ProcessLock,
    value: OpaqueTerm,
    arity: OpaqueTerm,
) -> ErlangResult {
    match value.into() {
        Term::Closure(fun) => match arity.into() {
            Term::Int(i) if i >= 0 => {
                return ErlangResult::Ok((fun.arity as i64 == i).into());
            }
            _ => {
                process.exception_info = ExceptionInfo::error(atoms::Badarg.into());
                process.exception_info.value = arity;
                return ErlangResult::Err;
            }
        },
        _ => {
            process.exception_info = ExceptionInfo::error(atoms::Badarg.into());
            process.exception_info.value = value;
            return ErlangResult::Err;
        }
    }
}

#[export_name = "erlang:is_big_integer/1"]
pub extern "C" fn is_big_integer1(_process: &mut ProcessLock, value: OpaqueTerm) -> ErlangResult {
    Ok(intrinsics::is_big_integer(value).into()).into()
}

#[export_name = "erlang:is_record/2"]
pub extern "C" fn is_record2(
    _process: &mut ProcessLock,
    value: OpaqueTerm,
    tag: OpaqueTerm,
) -> ErlangResult {
    match value.into() {
        Term::Tuple(tuple) => ErlangResult::Ok(tuple[0].eq(&tag).into()),
        _ => ErlangResult::Ok(false.into()),
    }
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
