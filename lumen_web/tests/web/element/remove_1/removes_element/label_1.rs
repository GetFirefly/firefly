use std::convert::TryInto;
use std::sync::Arc;

use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::{code, Process};
use liblumen_alloc::erts::term::{atom_unchecked, resource, Boxed, Tuple};
use liblumen_alloc::ModuleFunctionArity;

use web_sys::Window;

use super::label_2;

pub fn place_frame(process: &Process, placement: Placement) {
    process.place_frame(frame(), placement);
}

// Private

// ```elixir
// # label 1
// # pushed to stack: ()
// # returned from call: {:ok, window}
// # full stack: ({:ok, window})
// # returns: {:ok, document}
// {:ok, document} = Lumen.Web.Window.document(window)
// {:ok, body} = Lumen.Web.Document.body(document)
// {:ok, child} = Lumen.Web.Document.create_element(body, "table");
// :ok = Lumen.Web.Node.append_child(document, child);
// :ok = Lumen.Web.Element.remove(child);
// Lumen.Web.Wait.with_return(body_tuple)
// ```
fn code(arc_process: &Arc<Process>) -> code::Result {
    arc_process.reduce();

    let ok_window = arc_process.stack_pop().unwrap();
    assert!(
        ok_window.is_tuple(),
        "ok_window ({:?}) is not a tuple",
        ok_window
    );
    let ok_window_tuple: Boxed<Tuple> = ok_window.try_into().unwrap();
    assert_eq!(ok_window_tuple.len(), 2);
    assert_eq!(ok_window_tuple[0], atom_unchecked("ok"));
    let window = ok_window_tuple[1];
    let window_reference: resource::Reference = window.try_into().unwrap();
    let _: &Window = window_reference.downcast_ref().unwrap();

    label_2::place_frame(arc_process, Placement::Replace);
    lumen_web::window::document_1::place_frame_with_arguments(
        arc_process,
        Placement::Push,
        window,
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
