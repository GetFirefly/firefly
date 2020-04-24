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
) -> Result<(), Alloc> {
    assert!(document.is_boxed_resource_reference());
    process.stack_push(document)?;
    process.place_frame(frame(), placement);

    Ok(())
}

// Private

// ```elixir
// # label 3
// # pushed to stack: (document)
// # returned from call: {:ok, body}
// # full stack: ({:ok, body}, document)
// # returns: {:ok, child}
// {:ok, child} = Lumen.Web.Document.create_element(document, "table");
// :ok = Lumen.Web.Node.append_child(document, child);
// remove_ok = Lumen.Web.Element.remove(child);
// Lumen.Web.Wait.with_return(remove_ok)
// ```
fn code(arc_process: &Arc<Process>) -> frames::Result {
    arc_process.reduce();

    let ok_body = arc_process.stack_pop().unwrap();
    assert!(
        ok_body.is_boxed_tuple(),
        "ok_body ({:?}) is not a tuple",
        ok_body
    );
    let ok_body_tuple: Boxed<Tuple> = ok_body.try_into().unwrap();
    assert_eq!(ok_body_tuple.len(), 2);
    assert_eq!(ok_body_tuple[0], Atom::str_to_term("ok"));
    let body = ok_body_tuple[1];
    assert!(body.is_boxed_resource_reference());

    let document = arc_process.stack_pop().unwrap();
    assert!(document.is_boxed_resource_reference());

    label_4::place_frame_with_arguments(arc_process, Placement::Replace, body)?;

    let child_tag = arc_process.binary_from_str("table")?;
    liblumen_web::document::create_element_2::place_frame_with_arguments(
        arc_process,
        Placement::Push,
        document,
        child_tag,
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
