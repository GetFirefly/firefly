use std::convert::TryInto;
use std::sync::Arc;

use liblumen_alloc::erts::exception::system::Alloc;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::{code, Process};
use liblumen_alloc::erts::term::{atom_unchecked, resource, Boxed, Term, Tuple};
use liblumen_alloc::ModuleFunctionArity;

use web_sys::Element;

use super::label_5;

pub fn place_frame_with_arguments(
    process: &Process,
    placement: Placement,
    body: Term,
) -> Result<(), Alloc> {
    assert!(body.is_resource_reference());
    process.stack_push(body)?;
    process.place_frame(frame(), placement);

    Ok(())
}

// Private

// ```elixir
// # label 3
// # pushed to stack: (body)
// # returned from call: {:ok, child}
// # full stack: ({:ok, child}, body)
// # returns: :ok
// :ok = Lumen.Web.Node.append_child(body, child);
// remove_ok = Lumen.Web.Element.remove(child);
// Lumen.Web.Wait.with_return(remove_ok)
// ```
fn code(arc_process: &Arc<Process>) -> code::Result {
    arc_process.reduce();

    let ok_child = arc_process.stack_pop().unwrap();
    assert!(
        ok_child.is_tuple(),
        "ok_child ({:?}) is not a tuple",
        ok_child
    );
    let ok_child_tuple: Boxed<Tuple> = ok_child.try_into().unwrap();
    assert_eq!(ok_child_tuple.len(), 2);
    assert_eq!(ok_child_tuple[0], atom_unchecked("ok"));
    let child = ok_child_tuple[1];
    let child_reference: resource::Reference = child.try_into().unwrap();
    let _: &Element = child_reference.downcast_ref().unwrap();

    let body = arc_process.stack_pop().unwrap();
    assert!(body.is_resource_reference());

    label_5::place_frame_with_arguments(arc_process, Placement::Replace, child)?;

    lumen_web::node::append_child_2::place_frame_with_arguments(
        arc_process,
        Placement::Push,
        body,
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
