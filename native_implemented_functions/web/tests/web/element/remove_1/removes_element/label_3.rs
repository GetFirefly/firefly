//! ```elixir
//! # label 3
//! # pushed to stack: (document)
//! # returned from call: {:ok, body}
//! # full stack: ({:ok, body}, document)
//! # returns: {:ok, child}
//! {:ok, child} = Lumen.Web.Document.create_element(document, "table");
//! :ok = Lumen.Web.Node.append_child(document, child);
//! remove_ok = Lumen.Web.Element.remove(child);
//! Lumen.Web.Wait.with_return(remove_ok)
//! ```

use std::convert::TryInto;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::{Frame, Native, Process};
use liblumen_alloc::erts::term::prelude::*;

use liblumen_web::runtime::process::current_process;

use super::label_4;

pub fn frame() -> Frame {
    super::frame(NATIVE)
}

// Private

const NATIVE: Native = Native::Two(native);

extern "C" fn native(ok_body: Term, document: Term) -> Term {
    let arc_process = current_process();
    arc_process.reduce();

    arc_process.return_status(result(&arc_process, ok_body, document))
}

fn result(process: &Process, ok_body: Term, document: Term) -> exception::Result<Term> {
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

    assert!(document.is_boxed_resource_reference());

    process.queue_frame_with_arguments(label_4::frame().with_arguments(true, &[body]));

    let child_tag = process.binary_from_str("table")?;
    process.queue_frame_with_arguments(
        liblumen_web::document::create_element_2::frame()
            .with_arguments(false, &[document, child_tag]),
    );

    Ok(Term::NONE)
}
