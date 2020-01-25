use liblumen_alloc::erts::exception::InternalResult;
use liblumen_alloc::erts::term::closure::Definition;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::erts::Process;

use crate::code;

use super::{atom, small_integer};

pub fn decode<'a>(
    process: &Process,
    safe: bool,
    bytes: &'a [u8],
) -> InternalResult<(Term, &'a [u8])> {
    let (module, after_module_bytes) = atom::decode_tagged(safe, bytes)?;
    let (function, after_function_bytes) = atom::decode_tagged(safe, after_module_bytes)?;
    let (arity, after_arity_bytes) = small_integer::decode_tagged_u8(after_function_bytes)?;
    let definition = Definition::Export { function };

    let option_located_code = code::get(&module, &definition, arity);

    let closure =
        process.closure_with_env_from_slice(module, definition, arity, option_located_code, &[])?;

    Ok((closure, after_arity_bytes))
}
