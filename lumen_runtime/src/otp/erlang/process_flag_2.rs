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
use liblumen_alloc::erts::term::{Atom, Term};
use liblumen_alloc::{badarg, ModuleFunctionArity};

pub fn place_frame_with_arguments(
    process: &ProcessControlBlock,
    placement: Placement,
    flag: Term,
    value: Term,
) -> Result<(), Alloc> {
    process.stack_push(value)?;
    process.stack_push(flag)?;
    process.place_frame(frame(), placement);

    Ok(())
}

// Private

fn code(arc_process: &Arc<ProcessControlBlock>) -> code::Result {
    arc_process.reduce();

    let flag = arc_process.stack_pop().unwrap();
    let value = arc_process.stack_pop().unwrap();

    match native(arc_process, flag, value) {
        Ok(old_value) => {
            arc_process.return_from_call(old_value)?;

            ProcessControlBlock::call_code(arc_process)
        }
        Err(exception) => result_from_exception(arc_process, exception),
    }
}

fn frame() -> Frame {
    Frame::new(module_function_arity(), code)
}

fn function() -> Atom {
    Atom::try_from_str("process_flag").unwrap()
}

fn module_function_arity() -> Arc<ModuleFunctionArity> {
    Arc::new(ModuleFunctionArity {
        module: super::module(),
        function: function(),
        arity: 2,
    })
}

fn native(process: &ProcessControlBlock, flag: Term, value: Term) -> exception::Result {
    let flag_atom: Atom = flag.try_into()?;

    match flag_atom.name() {
        "error_handler" => unimplemented!(),
        "max_heap_size" => unimplemented!(),
        "message_queue_data" => unimplemented!(),
        "min_bin_vheap_size" => unimplemented!(),
        "min_heap_size" => unimplemented!(),
        "priority" => unimplemented!(),
        "save_calls" => unimplemented!(),
        "sensitive" => unimplemented!(),
        "trap_exit" => {
            let value_bool: bool = value.try_into()?;

            Ok(process.trap_exit(value_bool).into())
        }
        _ => Err(badarg!().into()),
    }
}
