use std::sync::Arc;

use liblumen_alloc::erts::exception::Alloc;
use liblumen_alloc::erts::process::frames::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::{frames, Process};
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::ModuleFunctionArity;

pub fn place_frame_with_arguments(
    process: &Process,
    placement: Placement,
    parent: Term,
    old_child: Term,
) -> Result<(), Alloc> {
    process.stack_push(old_child)?;
    process.stack_push(parent)?;
    process.place_frame(frame(), placement);

    Ok(())
}

// Private

// ```elixir
// # label 4
// # pushed to stack: (parent. old_child)
// # returned form call: :ok
// # full stack: (:ok, parent, old_child)
// # returns: {:error, :hierarchy_request}
// {:error, :hierarchy_request} = Lumen.Web.replace_child(parent, old_child, parent)
// ```
fn code(arc_process: &Arc<Process>) -> frames::Result {
    arc_process.reduce();

    let ok = arc_process.stack_pop().unwrap();
    assert_eq!(ok, Atom::str_to_term("ok"));
    let parent = arc_process.stack_pop().unwrap();
    assert!(parent.is_boxed_resource_reference());
    let old_child = arc_process.stack_pop().unwrap();
    assert!(old_child.is_boxed_resource_reference());

    liblumen_web::node::replace_child_3::place_frame_with_arguments(
        arc_process,
        Placement::Replace,
        parent,
        parent,
        old_child,
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
