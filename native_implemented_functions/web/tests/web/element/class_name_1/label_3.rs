use std::convert::TryInto;
use std::sync::Arc;

use liblumen_alloc::erts::process::frames::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::{frames, Process};
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::ModuleFunctionArity;

pub fn place_frame(process: &Process, placement: Placement) {
    process.place_frame(frame(), placement);
}

// Private

// ```elixir
// # label 3
// # pushed to stack: ()
// # returned from call: {:ok, body}
// # full stack: ({:ok, body})
// # returns: class_name
// class_name = Lumen.Web.Element.class_name(body)
// Lumen.Web.Wait.with_return(class_name)
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

    liblumen_web::element::class_name_1::place_frame_with_arguments(
        arc_process,
        Placement::Replace,
        body,
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
