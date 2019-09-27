// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;

use liblumen_alloc::badarg;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::{Boxed, Term, Tuple, TypedTerm};

use liblumen_alloc::HeapAlloc;
use lumen_runtime_macros::native_implemented_function;

#[native_implemented_function(make_tuple/3)]
pub fn native(
    process: &Process,
    arity: Term,
    default_value: Term,
    init_list: Term,
) -> exception::Result {
    // arity by definition is only 0-225, so `u8`, but ...
    let arity_u8: u8 = arity.try_into()?;
    // ... everything else uses `usize`, so cast it back up
    let arity_usize: usize = arity_u8 as usize;

    let mut heap = process.acquire_heap();
    let tuple = heap.mut_tuple(arity_usize)?;

    for index in 0..arity_usize {
        tuple
            .set_element_from_zero_based_usize_index(index, default_value)
            .unwrap();
    }

    match init_list.to_typed_term().unwrap() {
        TypedTerm::Nil => Ok(Term::make_boxed(tuple as *const Tuple)),
        TypedTerm::List(boxed_cons) => {
            for result in boxed_cons.into_iter() {
                match result {
                    Ok(init) => {
                        let init_boxed_tuple: Boxed<Tuple> = init.try_into()?;

                        if init_boxed_tuple.len() == 2 {
                            let index = init_boxed_tuple[0];
                            let element = init_boxed_tuple[1];
                            tuple.set_element_from_one_based_term_index(index, element)?;
                        } else {
                            return Err(badarg!().into());
                        }
                    }
                    Err(_) => return Err(badarg!().into()),
                }
            }

            Ok(Term::make_boxed(tuple as *const Tuple))
        }
        _ => Err(badarg!().into()),
    }
}
