use crate::function::ErlangResult;
use crate::term::{OpaqueTerm, Term, TermType};

/// This is an intrinsic expected by the compiler to be defined as part of the runtime, and is used for runtime type checking
#[export_name = "__lumen_builtin_typeof"]
pub extern "C" fn r#typeof(value: OpaqueTerm) -> TermType {
    value.r#typeof()
}

/// This is an intrinsic expected by the compiler to be defined as part of the runtime, and is used for runtime type checking
#[export_name = "__lumen_builtin_is_atom"]
pub extern "C" fn is_atom(value: OpaqueTerm) -> bool {
    value.is_atom()
}

/// This is an intrinsic expected by the compiler to be defined as part of the runtime, and is used for runtime type checking
#[export_name = "__lumen_builtin_is_number"]
pub extern "C" fn is_number(value: OpaqueTerm) -> bool {
    value.is_number()
}

/// This is an intrinsic expected by the compiler to be defined as part of the runtime, and is used for runtime type checking
#[export_name = "__lumen_builtin_is_tuple"]
#[allow(improper_ctypes_definitions)]
pub extern "C" fn is_tuple(value: OpaqueTerm) -> Result<u32, u32> {
    match value.into() {
        Term::Tuple(tup) => Ok(unsafe { tup.as_ref().len() as u32 }),
        _ => Err(0),
    }
}

/// This is an intrinsic expected by the compiler to be defined as part of the runtime
#[export_name = "__lumen_builtin_size"]
pub extern "C" fn size(value: OpaqueTerm) -> usize {
    value.size()
}

#[export_name = "erlang:is_atom/1"]
#[allow(improper_ctypes_definitions)]
pub extern "C" fn is_atom1(value: OpaqueTerm) -> ErlangResult {
    Ok(value.is_atom().into())
}

#[export_name = "erlang:is_list/1"]
#[allow(improper_ctypes_definitions)]
pub extern "C" fn is_list1(value: OpaqueTerm) -> ErlangResult {
    Ok(value.is_list().into())
}

#[export_name = "erlang:is_binary/1"]
#[allow(improper_ctypes_definitions)]
pub extern "C" fn is_binary1(value: OpaqueTerm) -> ErlangResult {
    Ok((value.r#typeof() == TermType::Binary).into())
}
