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
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::{Atom, Map, Term};
use liblumen_alloc::{badarg, ModuleFunctionArity};

pub fn place_frame_with_arguments(
    process: &Process,
    placement: Placement,
    list: Term,
) -> Result<(), Alloc> {
    process.stack_push(list)?;
    process.place_frame(frame(), placement);

    Ok(())
}

// Crate Public

pub(in crate::otp) fn code(arc_process: &Arc<Process>) -> code::Result {
    arc_process.reduce();

    let list = arc_process.stack_pop().unwrap();

    match native(arc_process, list) {
        Ok(map) => {
            arc_process.return_from_call(map)?;

            Process::call_code(arc_process)
        }
        Err(exception) => result_from_exception(arc_process, exception),
    }
}

// Private

fn frame() -> Frame {
    Frame::new(module_function_arity(), code)
}

fn function() -> Atom {
    Atom::try_from_str("from_list").unwrap()
}

fn module_function_arity() -> Arc<ModuleFunctionArity> {
    Arc::new(ModuleFunctionArity {
        module: super::module(),
        function: function(),
        arity: 1,
    })
}

fn native(process: &Process, list: Term) -> exception::Result {
    match Map::from_list(list) {
        Some(hash_map) => Ok(process.map_from_hash_map(hash_map)?),
        None => Err(badarg!().into()),
    }
}
