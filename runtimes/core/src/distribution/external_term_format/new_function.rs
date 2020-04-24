use std::convert::TryInto;
use std::ffi::c_void;
use std::mem;

use liblumen_alloc::erts::apply::find_symbol;
use liblumen_alloc::erts::exception::InternalResult;
use liblumen_alloc::erts::term::closure::{Definition, OldUnique};
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::erts::Process;
use liblumen_alloc::ModuleFunctionArity;

use super::{atom, decode_vec_term, isize, u32, u8, Pid};
use crate::distribution::external_term_format::try_split_at;

pub fn decode<'a>(
    process: &Process,
    safe: bool,
    bytes: &'a [u8],
) -> InternalResult<(Term, &'a [u8])> {
    let (total_byte_len, after_size_bytes) = u32::decode(bytes)?;
    let (arity, after_arity_bytes) = u8::decode(after_size_bytes)?;
    let (uniq, after_uniq_bytes) = decode_uniq(after_arity_bytes)?;
    let (index, after_index_bytes) = u32::decode(after_uniq_bytes)?;
    let (num_free, after_num_free_bytes) = u32::decode(after_index_bytes)?;
    let (module, after_module_bytes) = atom::decode_tagged(safe, after_num_free_bytes)?;

    let (old_index, after_old_index_bytes) = isize::decode(after_module_bytes)?;
    assert_eq!(old_index, index as isize);

    let (old_uniq, after_old_uniq_bytes) = isize::decode(after_old_index_bytes)?;
    let old_unique = old_uniq as OldUnique;

    let (creator, after_creator_bytes) = Pid::decode(safe, after_old_uniq_bytes)?;

    let env_len: usize = num_free as usize;
    let (env_vec, after_vec_term_bytes) =
        decode_vec_term(process, safe, after_creator_bytes, env_len)?;

    assert_eq!(
        bytes.len() - after_vec_term_bytes.len(),
        total_byte_len as usize
    );

    let definition = Definition::Anonymous {
        index: index as usize,
        unique: uniq,
        old_unique,
    };

    let module_function_arity = ModuleFunctionArity {
        module,
        function: definition.function_name(),
        arity,
    };

    let option_native = find_symbol(&module_function_arity)
        .map(|dynamic_callee| unsafe { mem::transmute::<_, *const c_void>(dynamic_callee) });

    let closure = process.anonymous_closure_with_env_from_slice(
        module,
        index,
        old_unique,
        uniq,
        arity,
        option_native,
        creator.into(),
        &env_vec,
    )?;

    Ok((closure, after_vec_term_bytes))
}

const UNIQ_LEN: usize = 16;

fn decode_uniq(bytes: &[u8]) -> InternalResult<([u8; UNIQ_LEN], &[u8])> {
    try_split_at(bytes, UNIQ_LEN).map(|(uniq_bytes, after_uniq_bytes)| {
        let uniq_array = uniq_bytes.try_into().unwrap();

        (uniq_array, after_uniq_bytes)
    })
}
