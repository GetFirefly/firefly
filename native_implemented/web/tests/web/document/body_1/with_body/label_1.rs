//! ```elixir
//! # label 1
//! # pushed to stack: ()
//! # returned from call: {:ok, window}
//! # full stack: ({:ok, window})
//! # returns: {:ok, document}
//! {:ok, document} = Lumen.Web.Window.document(window)
//! body_tuple = Lumen.Web.Document.body(document)
//! Lumen.Web.Wait.with_return(body_tuple)
//! ```

use std::convert::TryInto;

use liblumen_rt::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use super::label_2;

#[native_implemented::label]
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
    assert!(window.is_boxed_resource_reference());

    process.queue_frame_with_arguments(
        liblumen_web::window::document_1::frame().with_arguments(false, &[window]),
    );
    process.queue_frame_with_arguments(label_2::frame().with_arguments(true, &[]));

    Term::NONE
}
