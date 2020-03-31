// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;

use anyhow::*;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::alloc::TermAlloc;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use native_implemented_function::native_implemented_function;

use crate::runtime::context::*;

#[native_implemented_function(make_tuple/3)]
pub fn native(
    process: &Process,
    arity: Term,
    default_value: Term,
    init_list: Term,
) -> exception::Result<Term> {
    // arity by definition is only 0-225, so `u8`, but ...
    let arity_u8 = term_try_into_arity(arity)?;
    // ... everything else uses `usize`, so cast it back up
    let arity_usize: usize = arity_u8 as usize;

    let mut heap = process.acquire_heap();
    let mut tuple = heap.mut_tuple(arity_usize)?;

    for index in 0..arity_usize {
        tuple.set_element(index, default_value).unwrap();
    }

    match init_list.decode().unwrap() {
        TypedTerm::Nil => Ok(tuple.encode()?),
        TypedTerm::List(boxed_cons) => {
            for result in boxed_cons.into_iter() {
                match result {
                    Ok(init) => {
                        let init_boxed_tuple: Boxed<Tuple> = init.try_into().with_context(|| format!("init list ({}) element ({}) is not {{position :: pos_integer(), term()}}", init_list, init))?;

                        if init_boxed_tuple.len() == 2 {
                            let position = init_boxed_tuple[0];
                            let index: OneBasedIndex = position.try_into().with_context(|| {
                                format!("init list ({}) element ({}) position ({}) is not a positive integer", init_list, init, position)
                            })?;

                            let element = init_boxed_tuple[1];
                            tuple.set_element(index, element).with_context(|| {
                                format!("position ({}) cannot be set", position)
                            })?;
                        } else {
                            return Err(anyhow!(
                                "init list ({}) element ({}) is a tuple, but not 2-arity",
                                init_list,
                                init
                            )
                            .into());
                        }
                    }
                    Err(_) => {
                        return Err(ImproperListError)
                            .context(format!("init_list ({}) is improper", init_list))
                            .map_err(From::from)
                    }
                }
            }

            Ok(tuple.encode()?)
        }
        _ => Err(TypeError)
            .context(format!("init_list ({}) is not a list", init_list))
            .map_err(From::from),
    }
}
