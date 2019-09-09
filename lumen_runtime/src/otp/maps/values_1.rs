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
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::{Atom, Boxed, Map, Term};
use liblumen_alloc::{badmap, ModuleFunctionArity};

pub fn place_frame_with_arguments(
    process: &Process,
    placement: Placement,
    map: Term,
) -> Result<(), Alloc> {
    process.stack_push(map)?;
    process.place_frame(frame(), placement);

    Ok(())
}

// Crate Public

pub(in crate::otp) fn code(arc_process: &Arc<Process>) -> code::Result {
    arc_process.reduce();

    let map = arc_process.stack_pop().unwrap();

    match native(arc_process, map) {
        Ok(values) => {
            arc_process.return_from_call(values)?;

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
    Atom::try_from_str("values").unwrap()
}

fn module_function_arity() -> Arc<ModuleFunctionArity> {
    Arc::new(ModuleFunctionArity {
        module: super::module(),
        function: function(),
        arity: 1,
    })
}

fn native(process: &Process, map: Term) -> exception::Result {
    let result_map: Result<Boxed<Map>, _> = map.try_into();

    match result_map {
        Ok(map) => {
            let values = map.values();
            let list = process.list_from_slice(&values)?;

            Ok(list)
        }

        Err(_) => Err(badmap!(process, map)),
    }
}
