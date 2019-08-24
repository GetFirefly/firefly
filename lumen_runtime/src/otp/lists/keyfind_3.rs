// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::sync::Arc;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::exception::system::Alloc;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::code::{self, result_from_exception};
use liblumen_alloc::erts::process::ProcessControlBlock;
use liblumen_alloc::erts::term::{Atom, Term};
use liblumen_alloc::ModuleFunctionArity;

use crate::otp::lists::get_by_term_one_based_index_key;

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

pub fn native(key: Term, one_based_index: Term, tuple_list: Term) -> exception::Result {
    get_by_term_one_based_index_key(tuple_list, one_based_index, key).map(|option| match option {
        Some(found) => found,
        None => false.into(),
    })
}
