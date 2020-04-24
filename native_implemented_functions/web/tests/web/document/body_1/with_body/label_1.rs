use std::convert::TryInto;
use std::sync::Arc;

use liblumen_alloc::erts::process::frames::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::{frames, Process};
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::ModuleFunctionArity;

use super::label_2;

pub fn place_frame(process: &Process, placement: Placement) {
    process.place_frame(frame(), placement);
}

// Private

/// ```elixir
/// # label 1
/// # pushed to stack: ()
/// # returned from call: {:ok, window}
/// # full stack: ({:ok, window})
/// # returns: {:ok, document}
/// {:ok, document} = Lumen.Web.Window.document(window)
/// body_tuple = Lumen.Web.Document.body(document)
/// Lumen.Web.Wait.with_return(body_tuple)
/// ```
fn code(arc_process: &Arc<Process>) -> frames::Result {
    arc_process.reduce();

    let ok_window = arc_process.stack_pop().unwrap();
    assert!(
        ok_window.is_boxed_tuple(),
        "ok_window ({:?}) is not a tuple",
        ok_window
    );
    let ok_window_tuple: Boxed<Tuple> = ok_window.try_into().unwrap();
    assert_eq!(ok_window_tuple.len(), 2);
    assert_eq!(ok_window_tuple[0], Atom::str_to_term("ok"));
    let window = ok_window_tuple[1];
    assert!(window.is_boxed_resource_reference());

    label_2::place_frame(arc_process, Placement::Replace);
    liblumen_web::window::document_1::place_frame_with_arguments(
        arc_process,
        Placement::Push,
        window,
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
