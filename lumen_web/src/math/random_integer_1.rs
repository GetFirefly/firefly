use std::convert::TryInto;
use std::sync::Arc;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::exception::system::Alloc;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::code::{self, result_from_exception};
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::erts::ModuleFunctionArity;

/// Generates an integer between 0 and (exclusive_max - 1).
///
/// ```elixir
/// random_integer = Lumen.Web.Math.random_integer(exclusive_max)
/// ```
pub fn place_frame_with_arguments(
    process: &Process,
    placement: Placement,
    exclusive_max: Term,
) -> Result<(), Alloc> {
    process.stack_push(exclusive_max)?;
    process.place_frame(frame(), placement);

    Ok(())
}

// Private

fn code(arc_process: &Arc<Process>) -> code::Result {
    arc_process.reduce();

    let exclusive_max = arc_process.stack_pop().unwrap();

    match native(arc_process, exclusive_max) {
        Ok(random_integer) => {
            arc_process.return_from_call(random_integer)?;

            Process::call_code(arc_process)
        }
        Err(exception) => result_from_exception(arc_process, exception),
    }
}

fn frame() -> Frame {
    Frame::new(module_function_arity(), code)
}

fn function() -> Atom {
    Atom::try_from_str("random_integer").unwrap()
}

fn module_function_arity() -> Arc<ModuleFunctionArity> {
    Arc::new(ModuleFunctionArity {
        module: super::module(),
        function: function(),
        arity: 1,
    })
}

fn native(process: &Process, exclusive_max: Term) -> exception::Result {
    let exclusive_max_usize: usize = exclusive_max.try_into()?;
    let exclusive_max_f64 = exclusive_max_usize as f64;
    let random_usize = (js_sys::Math::random() * exclusive_max_f64).trunc() as usize;

    process.integer(random_usize).map_err(|error| error.into())
}
