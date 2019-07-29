use std::convert::TryInto;
use std::sync::Arc;

use liblumen_alloc::erts::exception::runtime;
use liblumen_alloc::erts::process::code;
use liblumen_alloc::erts::process::code::stack::frame::Frame;
use liblumen_alloc::erts::process::ProcessControlBlock;
use liblumen_alloc::erts::term::{atom_unchecked, Atom};
use liblumen_alloc::erts::ModuleFunctionArity;

pub fn puts_frame() -> Frame {
    let module_function_arity = Arc::new(ModuleFunctionArity {
        module: Atom::try_from_str("Elixir.IO").unwrap(),
        function: Atom::try_from_str("puts").unwrap(),
        arity: 1,
    });

    Frame::new(module_function_arity, puts_code)
}

fn puts_code(arc_process: &Arc<ProcessControlBlock>) -> code::Result {
    let elixir_string = arc_process.stack_pop().unwrap();

    match elixir_string.try_into(): Result<String, runtime::Exception> {
        Ok(string) => {
            // NOT A DEBUGGING LOG
            crate::start::log_1(string);
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
