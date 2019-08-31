// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;
use std::sync::Arc;

use num_bigint::BigInt;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::exception::system::Alloc;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::code::{self, result_from_exception};
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::{Atom, Term};
use liblumen_alloc::{badarg, ModuleFunctionArity};

pub fn place_frame_with_arguments(
    process: &Process,
    placement: Placement,
    binary: Term,
) -> Result<(), Alloc> {
    process.stack_push(binary)?;
    process.place_frame(frame(), placement);

    Ok(())
}

// Private

fn code(arc_process: &Arc<Process>) -> code::Result {
    arc_process.reduce();

    let binary = arc_process.stack_pop().unwrap();

    match native(arc_process, binary) {
        Ok(boolean) => {
            arc_process.return_from_call(boolean)?;

            Process::call_code(arc_process)
        }
        Err(exception) => result_from_exception(arc_process, exception),
    }
}

fn frame() -> Frame {
    Frame::new(module_function_arity(), code)
}

fn function() -> Atom {
    Atom::try_from_str("binary_to_integer").unwrap()
}

fn module_function_arity() -> Arc<ModuleFunctionArity> {
    Arc::new(ModuleFunctionArity {
        module: super::module(),
        function: function(),
        arity: 1,
    })
}

fn native(process: &Process, binary: Term) -> exception::Result {
    let string: String = binary.try_into()?;

    match BigInt::parse_bytes(string.as_bytes(), 10) {
        Some(big_int) => process.integer(big_int).map_err(|error| error.into()),
        None => Err(badarg!().into()),
    }
}
