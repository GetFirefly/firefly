use std::convert::TryInto;
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
) -> Result<(), Alloc> {
    process.stack_push(parent)?;
    process.place_frame(frame(), placement);

    Ok(())
}

// Private

// ```elixir
// # label 6
// # pushed to stack: (parent)
// # returned form call: {:ok, new_child}
// # full stack: ({:ok, new_child}, parent)
// # returns: {:ok, inserted_child}
// {:ok, inserted_child} = Lumen.Web.insert_before(parent, new_child, nil)
// ```
fn code(arc_process: &Arc<Process>) -> frames::Result {
    arc_process.reduce();

    let ok_new_child = arc_process.stack_pop().unwrap();
    assert!(
        ok_new_child.is_boxed_tuple(),
        "ok_new_child ({:?}) is not a tuple",
        ok_new_child
    );
    let ok_new_child_tuple: Boxed<Tuple> = ok_new_child.try_into().unwrap();
    assert_eq!(ok_new_child_tuple.len(), 2);
    assert_eq!(ok_new_child_tuple[0], Atom::str_to_term("ok"));
    let new_child = ok_new_child_tuple[1];
    assert!(new_child.is_boxed_resource_reference());

    let parent = arc_process.stack_pop().unwrap();
    assert!(parent.is_boxed_resource_reference());

    let reference_child = Atom::str_to_term("nil");
    liblumen_web::node::insert_before_3::place_frame_with_arguments(
        arc_process,
        Placement::Replace,
        parent,
        new_child,
        reference_child,
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
