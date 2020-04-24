use std::sync::Arc;

use liblumen_alloc::erts::exception::Alloc;
use liblumen_alloc::erts::process::frames::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::{frames, Process};
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::ModuleFunctionArity;

pub fn place_frame_with_arguments(
    process: &Process,
    placement: Placement,
    child: Term,
) -> Result<(), Alloc> {
    assert!(child.is_boxed_resource_reference());
    process.stack_push(child)?;
    process.place_frame(frame(), placement);

    Ok(())
}

// Private

// ```elixir
// # label 5
// # pushed to stack: (child)
// # returned from call: :ok
// # full stack: (:ok, child)
// # returns: :ok
// remove_ok = Lumen.Web.Element.remove(child);
// Lumen.Web.Wait.with_return(remove_ok)
// ```
fn code(arc_process: &Arc<Process>) -> frames::Result {
    arc_process.reduce();

    let ok = arc_process.stack_pop().unwrap();
    assert_eq!(ok, Atom::str_to_term("ok"));

    let child = arc_process.stack_pop().unwrap();
    assert!(child.is_boxed_resource_reference());

    liblumen_web::element::remove_1::place_frame_with_arguments(
        arc_process,
        Placement::Replace,
        child,
    )?;

    Process::call_native_or_yield(arc_process)
}

fn frame() -> Frame {
    let module_function_arity = Arc::new(ModuleFunctionArity {
        module: super::module(),
        function: super::function(),
        arity: 0,
    });

    Frame::new(module_function_arity, code)
}
