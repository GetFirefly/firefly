// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;
use std::sync::Arc;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::exception::system::Alloc;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::code::{self, result_from_exception};
use liblumen_alloc::erts::process::ProcessControlBlock;
use liblumen_alloc::erts::term::{Atom, Boxed, Cons, Term, Tuple, TypedTerm};
use liblumen_alloc::{badarg, ModuleFunctionArity};

use liblumen_alloc::erts::term::index::try_from_one_based_term_to_zero_based_usize;

pub fn place_frame_with_arguments(
    process: &ProcessControlBlock,
    placement: Placement,
    key: Term,
    one_based_index: Term,
    tuple_list: Term,
) -> Result<(), Alloc> {
    process.stack_push(tuple_list)?;
    process.stack_push(one_based_index)?;
    process.stack_push(key)?;
    process.place_frame(frame(), placement);

    Ok(())
}

// Private

fn code(arc_process: &Arc<ProcessControlBlock>) -> code::Result {
    arc_process.reduce();

    let key = arc_process.stack_pop().unwrap();
    let one_based_index = arc_process.stack_pop().unwrap();
    let tuple_list = arc_process.stack_pop().unwrap();

    match native(key, one_based_index, tuple_list) {
        Ok(tuple_or_false) => {
            arc_process.return_from_call(tuple_or_false)?;

            ProcessControlBlock::call_code(arc_process)
        }
        Err(exception) => result_from_exception(arc_process, exception),
    }
}

fn frame() -> Frame {
    Frame::new(module_function_arity(), code)
}

fn function() -> Atom {
    Atom::try_from_str("keyfind").unwrap()
}

fn module_function_arity() -> Arc<ModuleFunctionArity> {
    Arc::new(ModuleFunctionArity {
        module: super::module(),
        function: function(),
        arity: 3,
    })
}

fn native(key: Term, one_based_index: Term, tuple_list: Term) -> exception::Result {
    let index_zero_based_usize: usize =
        try_from_one_based_term_to_zero_based_usize(one_based_index)?;

    match tuple_list.to_typed_term().unwrap() {
        TypedTerm::Nil => Ok(false.into()),
        TypedTerm::List(cons) => native_from_cons(key, index_zero_based_usize, cons),
        _ => Err(badarg!().into()),
    }
}

fn native_from_cons(
    key: Term,
    index_zero_based_usize: usize,
    cons: Boxed<Cons>,
) -> exception::Result {
    for result in cons.into_iter() {
        match result {
            Ok(list_element) => {
                let list_element_result_tuple: Result<Boxed<Tuple>, _> = list_element.try_into();

                if let Ok(list_element_tuple) = list_element_result_tuple {
                    if let Ok(list_element_tuple_element) = list_element_tuple
                        .get_element_from_zero_based_usize_index(index_zero_based_usize)
                    {
                        if key == list_element_tuple_element {
                            return Ok(list_element);
                        }
                    }
                }
            }
            Err(_) => return Err(badarg!().into()),
        }
    }

    Ok(false.into())
}
