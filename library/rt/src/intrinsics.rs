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
#[export_name = "__lumen_builtin_is_list"]
pub extern "C" fn is_list(value: OpaqueTerm) -> bool {
    value.is_list()
}

/// This is an intrinsic expected by the compiler to be defined as part of the runtime, and is used for runtime type checking
#[allow(improper_ctypes_definitions)]
#[export_name = "__lumen_builtin_is_tuple"]
pub extern "C" fn is_tuple(value: OpaqueTerm) -> (bool, u32) {
    match value.into() {
        Term::Tuple(tup) => (true, unsafe { tup.as_ref().len() as u32 }),
        _ => (false, 0),
    }
}

/// This is an intrinsic expected by the compiler to be defined as part of the runtime
#[export_name = "__lumen_builtin_size"]
pub extern "C" fn size(value: OpaqueTerm) -> usize {
    value.size()
}
