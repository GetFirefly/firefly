use std::convert::TryInto;
use std::sync::Arc;

use liblumen_alloc::erts::exception::Alloc;
use liblumen_alloc::erts::process::frames::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::{frames, Process};
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::ModuleFunctionArity;

use super::label_4;

pub fn place_frame_with_arguments(
    process: &Process,
    placement: Placement,
    document: Term,
    reference_child: Term,
) -> Result<(), Alloc> {
    process.stack_push(reference_child)?;
    process.stack_push(document)?;
    process.place_frame(frame(), placement);

    Ok(())
}

// Private

// ```elixir
// # label 3
// # pushed to stack: (document, reference_child)
// # returned form call: {:ok, parent}
// # full stack: ({:ok, parent}, document, reference_child)
// # returns: :ok
// :ok = Lumen.Web.Node.append_child(document, parent)
// :ok = Lumen.Web.Node.append_child(parent, reference_child)
// {:ok, new_child} = Lumen.Web.Document.create_element(document, "ul");
// {:ok, inserted_child} = Lumen.Web.insert_before(parent, new_child, reference_child)
// ```
fn code(arc_process: &Arc<Process>) -> frames::Result {
    arc_process.reduce();

    let ok_parent = arc_process.stack_pop().unwrap();
    assert!(
        ok_parent.is_boxed_tuple(),
        "ok_parent ({:?}) is not a tuple",
        ok_parent
    );
    let ok_parent_tuple: Boxed<Tuple> = ok_parent.try_into().unwrap();
    assert_eq!(ok_parent_tuple.len(), 2);
    assert_eq!(ok_parent_tuple[0], Atom::str_to_term("ok"));
    let parent = ok_parent_tuple[1];
    assert!(parent.is_boxed_resource_reference());

    let document = arc_process.stack_pop().unwrap();
    assert!(document.is_boxed_resource_reference());

    let reference_child = arc_process.stack_pop().unwrap();
    assert!(reference_child.is_boxed_resource_reference());

    label_4::place_frame_with_arguments(
        arc_process,
        Placement::Replace,
        document,
        parent,
        reference_child,
    )?;

    liblumen_web::node::append_child_2::place_frame_with_arguments(
        arc_process,
        Placement::Push,
        document,
        parent,
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
