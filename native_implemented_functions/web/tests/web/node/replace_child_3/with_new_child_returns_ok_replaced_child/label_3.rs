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
    old_child: Term,
) -> Result<(), Alloc> {
    process.stack_push(old_child)?;
    process.stack_push(document)?;
    process.place_frame(frame(), placement);

    Ok(())
}

// Private

// ```elixir
// # label 3
// # pushed to stack: (document, old_child)
// # returned form call: {:ok, parent}
// # full stack: ({:ok, parent}, document, old_child)
// # returns: :ok
// :ok = Lumen.Web.Node.append_child(parent, old_child)
// {:ok, new_child} = Lumen.Web.Document.create_element(document, "ul");
// {:ok, replaced_child} = Lumen.Web.replace_child(parent, new_child, old_child)
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

    let old_child = arc_process.stack_pop().unwrap();
    assert!(old_child.is_boxed_resource_reference());

    label_4::place_frame_with_arguments(
        arc_process,
        Placement::Replace,
        document,
        parent,
        old_child,
    )?;

    liblumen_web::node::append_child_2::place_frame_with_arguments(
        arc_process,
        Placement::Push,
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
