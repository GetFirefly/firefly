use std::sync::Arc;

use liblumen_alloc::erts::exception::system::Alloc;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::{code, Process};
use liblumen_alloc::erts::term::{atom_unchecked, Term};
use liblumen_alloc::ModuleFunctionArity;

pub fn place_frame_with_arguments(
    process: &Process,
    placement: Placement,
    child: Term,
) -> Result<(), Alloc> {
    assert!(child.is_resource_reference());
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
fn code(arc_process: &Arc<Process>) -> code::Result {
    arc_process.reduce();

    let ok = arc_process.stack_pop().unwrap();
    assert_eq!(ok, atom_unchecked("ok"));

    let child = arc_process.stack_pop().unwrap();
    assert!(child.is_resource_reference());

    lumen_web::element::remove_1::place_frame_with_arguments(
        arc_process,
        Placement::Replace,
        child,
    )?;

    Process::call_code(arc_process)
}

fn frame() -> Frame {
    let module_function_arity = Arc::new(ModuleFunctionArity {
        module: super::module(),
        function: super::function(),
        arity: 0,
    });

    Frame::new(module_function_arity, code)
}
