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
use liblumen_alloc::ModuleFunctionArity;

use crate::time;

pub fn place_frame_with_arguments(
    process: &Process,
    placement: Placement,
    time: Term,
    from_unit: Term,
    to_unit: Term,
) -> Result<(), Alloc> {
    process.stack_push(to_unit)?;
    process.stack_push(from_unit)?;
    process.stack_push(time)?;
    process.place_frame(frame(), placement);

    Ok(())
}

// Private

fn code(arc_process: &Arc<Process>) -> code::Result {
    arc_process.reduce();

    let time = arc_process.stack_pop().unwrap();
    let from_unit = arc_process.stack_pop().unwrap();
    let to_unit = arc_process.stack_pop().unwrap();

    match native(arc_process, time, from_unit, to_unit) {
        Ok(child_pid) => {
            arc_process.return_from_call(child_pid)?;

            Process::call_code(arc_process)
        }
        Err(exception) => result_from_exception(arc_process, exception),
    }
}

fn frame() -> Frame {
    Frame::new(module_function_arity(), code)
}

fn function() -> Atom {
    Atom::try_from_str("spawn").unwrap()
}

fn module_function_arity() -> Arc<ModuleFunctionArity> {
    Arc::new(ModuleFunctionArity {
        module: super::module(),
        function: function(),
        arity: 3,
    })
}

pub fn native(process: &Process, time: Term, from_unit: Term, to_unit: Term) -> exception::Result {
    let time_big_int: BigInt = time.try_into()?;
    let from_unit_unit: time::Unit = from_unit.try_into()?;
    let to_unit_unit: time::Unit = to_unit.try_into()?;
    let converted_big_int = time::convert(time_big_int, from_unit_unit, to_unit_unit);
    let converted_term = process.integer(converted_big_int)?;

    Ok(converted_term)
}
