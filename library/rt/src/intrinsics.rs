///! The functions defined in this module are compiler builtins that are used to
///! implement runtime checks in a maximally efficient manner, i.e. invariants are
///! already guaranteed to be upheld by the compiler, and the result can be a native
///! machine type without the encoding/decoding overhead of terms.
#[cfg(feature = "async")]
use core::ptr::NonNull;

use crate::cmp::ExactEq;
use crate::term::{OpaqueTerm, Term, TermType};

#[inline]
#[export_name = "__firefly_builtin_typeof"]
pub extern "C" fn r#typeof(value: OpaqueTerm) -> TermType {
    value.r#typeof()
}

#[inline]
#[export_name = "__firefly_builtin_is_atom"]
pub extern "C" fn is_atom(value: OpaqueTerm) -> bool {
    value.is_atom()
}

#[inline]
#[export_name = "__firefly_builtin_is_list"]
pub extern "C" fn is_list(value: OpaqueTerm) -> bool {
    value.is_list()
}

#[inline]
#[export_name = "__firefly_builtin_is_binary"]
pub extern "C" fn is_binary(value: OpaqueTerm) -> bool {
    let term: Term = value.into();
    if let Some(bin) = term.as_bitstring() {
        bin.is_binary()
    } else {
        false
    }
}

#[inline]
#[export_name = "__firefly_builtin_is_big_integer"]
pub extern "C" fn is_big_integer(value: OpaqueTerm) -> bool {
    match value.into() {
        Term::BigInt(_) => true,
        _ => false,
    }
}

#[inline]
#[export_name = "__firefly_builtin_is_number"]
pub extern "C" fn is_number(value: OpaqueTerm) -> bool {
    value.is_number()
}

#[derive(Copy, Clone)]
#[repr(u8)]
pub enum IsTupleResult {
    Ok(u32) = 0,
    Err(u32) = 1,
}

#[inline]
#[export_name = "__firefly_builtin_is_tuple"]
pub extern "C" fn is_tuple(value: OpaqueTerm) -> IsTupleResult {
    value
        .tuple_size()
        .map(IsTupleResult::Ok)
        .unwrap_or(IsTupleResult::Err(0))
}

#[inline]
#[export_name = "__firefly_builtin_is_function"]
pub extern "C" fn is_function(value: OpaqueTerm) -> bool {
    value.r#typeof() == TermType::Closure
}

#[inline]
#[export_name = "__firefly_builtin_size"]
pub extern "C" fn size(value: OpaqueTerm) -> usize {
    value.size()
}

#[inline]
#[export_name = "__firefly_builtin_gte"]
pub extern "C" fn gte(lhs: OpaqueTerm, rhs: OpaqueTerm) -> bool {
    lhs >= rhs
}

#[inline]
#[export_name = "__firefly_builtin_gt"]
pub extern "C" fn gt(lhs: OpaqueTerm, rhs: OpaqueTerm) -> bool {
    lhs > rhs
}

#[inline]
#[export_name = "__firefly_builtin_lt"]
pub extern "C" fn lt(lhs: OpaqueTerm, rhs: OpaqueTerm) -> bool {
    lhs < rhs
}

#[inline]
#[export_name = "__firefly_builtin_lte"]
pub extern "C" fn lte(lhs: OpaqueTerm, rhs: OpaqueTerm) -> bool {
    lhs <= rhs
}

#[inline]
#[export_name = "__firefly_builtin_eq"]
pub extern "C" fn eq(lhs: OpaqueTerm, rhs: OpaqueTerm) -> bool {
    let lhs: Term = lhs.into();
    let rhs: Term = rhs.into();
    lhs == rhs
}

#[inline]
#[export_name = "__firefly_builtin_ne"]
pub extern "C" fn ne(lhs: OpaqueTerm, rhs: OpaqueTerm) -> bool {
    let lhs: Term = lhs.into();
    let rhs: Term = rhs.into();
    lhs != rhs
}

#[inline]
#[export_name = "__firefly_builtin_exact_eq"]
pub extern "C" fn exact_eq(lhs: OpaqueTerm, rhs: OpaqueTerm) -> bool {
    lhs.exact_eq(&rhs)
}

#[inline]
#[export_name = "__firefly_builtin_exact_ne"]
pub extern "C" fn exact_ne(lhs: OpaqueTerm, rhs: OpaqueTerm) -> bool {
    lhs.exact_ne(&rhs)
}
