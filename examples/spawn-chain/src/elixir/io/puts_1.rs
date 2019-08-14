use std::convert::TryInto;
use std::sync::Arc;

use liblumen_alloc::erts::exception::runtime;
use liblumen_alloc::erts::exception::system::Alloc;
use liblumen_alloc::erts::process::code;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::ProcessControlBlock;
use liblumen_alloc::erts::term::Term;
use liblumen_alloc::erts::term::{atom_unchecked, Atom};
use liblumen_alloc::erts::ModuleFunctionArity;

use lumen_runtime::system;

pub fn place_frame_with_arguments(
    process: &ProcessControlBlock,
    placement: Placement,
    binary: Term,
) -> Result<(), Alloc> {
    assert!(binary.is_binary());

    process.stack_push(binary)?;
    process.place_frame(frame(), placement);

    Ok(())
}

// Private

fn code(arc_process: &Arc<ProcessControlBlock>) -> code::Result {
    let elixir_string = arc_process.stack_pop().unwrap();

    match elixir_string.try_into(): Result<String, runtime::Exception> {
        Ok(string) => {
            // NOT A DEBUGGING LOG
            system::io::puts(&string);
            arc_process.reduce();

            let ok = atom_unchecked("ok");
            arc_process.return_from_call(ok)?;

            ProcessControlBlock::call_code(arc_process)
        }
        Err(exception) => {
            arc_process.reduce();
            arc_process.exception(exception);

            Ok(())
        }
    }
}

fn frame() -> Frame {
    let module_function_arity = Arc::new(ModuleFunctionArity {
        module: super::module(),
        function: function(),
        arity: 1,
    });

    Frame::new(module_function_arity, code)
}

fn function() -> Atom {
    Atom::try_from_str("puts").unwrap()
}
