use std::ffi::c_void;
use std::mem;

use liblumen_alloc::erts::exception::InternalResult;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::erts::Process;

use super::{atom, small_integer};
use liblumen_alloc::erts::apply::find_symbol;
use liblumen_alloc::ModuleFunctionArity;

pub fn decode<'a>(
    process: &Process,
    safe: bool,
    bytes: &'a [u8],
) -> InternalResult<(Term, &'a [u8])> {
    let (module, after_module_bytes) = atom::decode_tagged(safe, bytes)?;
    let (function, after_function_bytes) = atom::decode_tagged(safe, after_module_bytes)?;
    let (arity, after_arity_bytes) = small_integer::decode_tagged_u8(after_function_bytes)?;

    let module_function_arity = ModuleFunctionArity {
        module,
        function,
        arity,
    };

    let option_native = find_symbol(&module_function_arity)
        .map(|dynamic_callee| unsafe { mem::transmute::<_, *const c_void>(dynamic_callee) });

    let closure = process.export_closure(module, function, arity, option_native)?;

    Ok((closure, after_arity_bytes))
}
