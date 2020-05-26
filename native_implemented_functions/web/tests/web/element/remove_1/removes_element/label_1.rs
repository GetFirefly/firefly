//! ```elixir
//! # label 1
//! # pushed to stack: ()
//! # returned from call: {:ok, window}
//! # full stack: ({:ok, window})
//! # returns: {:ok, document}
//! {:ok, document} = Lumen.Web.Window.document(window)
//! {:ok, body} = Lumen.Web.Document.body(document)
//! {:ok, child} = Lumen.Web.Document.create_element(body, "table");
//! :ok = Lumen.Web.Node.append_child(document, child);
//! :ok = Lumen.Web.Element.remove(child);
//! Lumen.Web.Wait.with_return(body_tuple)
//! ```

use std::convert::TryInto;

use web_sys::Window;

use liblumen_alloc::erts::process::{Frame, Native, Process};
use liblumen_alloc::erts::term::prelude::*;

use liblumen_web::runtime::process::current_process;

use super::label_2;

pub fn frame() -> Frame {
    super::frame(NATIVE)
}

// Private

const NATIVE: Native = Native::One(native);

extern "C" fn native(ok_window: Term) -> Term {
    let arc_process = current_process();
    arc_process.reduce();

    result(&arc_process, ok_window)
}

fn result(process: &Process, ok_window: Term) -> Term {
    assert!(
        ok_window.is_boxed_tuple(),
        "ok_window ({:?}) is not a tuple",
        ok_window
    );
    let ok_window_tuple: Boxed<Tuple> = ok_window.try_into().unwrap();
    assert_eq!(ok_window_tuple.len(), 2);
    assert_eq!(ok_window_tuple[0], Atom::str_to_term("ok"));
    let window = ok_window_tuple[1];
    let window_ref_boxed: Boxed<Resource> = window.try_into().unwrap();
    let window_reference: Resource = window_ref_boxed.into();
    let _: &Window = window_reference.downcast_ref().unwrap();

    process.queue_frame_with_arguments(label_2::frame().with_arguments(true, &[]));
    process.queue_frame_with_arguments(
        liblumen_web::window::document_1::frame().with_arguments(false, &[window]),
    );

    Term::NONE
}
