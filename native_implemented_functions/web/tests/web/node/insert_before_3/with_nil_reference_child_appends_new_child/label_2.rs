//! ```elixir
//! # label 2
//! # pushed to stack: (document)
//! # returned form call: {:ok, existing_child}
//! # full stack: ({:ok, existing_child}, document)
//! # returns: {:ok, parent}
//! {:ok, parent} = Lumen.Web.Document.create_element(parent_document, "div")
//! :ok = Lumen.Web.Node.append_child(document, parent)
//! :ok = Lumen.Web.Node.append_child(parent, existing_child)
//! {:ok, new_child} = Lumen.Web.Document.create_element(document, "ul");
//! {:ok, inserted_child} = Lumen.Web.insert_before(parent, new_child, nil)
//! ```

use std::convert::TryInto;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::{Frame, Native, Process};
use liblumen_alloc::erts::term::prelude::*;

use liblumen_web::runtime::process::current_process;

use super::label_3;

pub fn frame() -> Frame {
    super::frame(NATIVE)
}

// Private

const NATIVE: Native = Native::Two(native);

extern "C" fn native(ok_existing_child: Term, document: Term) -> Term {
    let arc_process = current_process();
    arc_process.reduce();

    arc_process.return_status(result(&arc_process, ok_existing_child, document))
}

fn result(process: &Process, ok_existing_child: Term, document: Term) -> exception::Result<Term> {
    assert!(
        ok_existing_child.is_boxed_tuple(),
        "ok_existing_child ({:?}) is not a tuple",
        ok_existing_child
    );
    let ok_existing_child_tuple: Boxed<Tuple> = ok_existing_child.try_into().unwrap();
    assert_eq!(ok_existing_child_tuple.len(), 2);
    assert_eq!(ok_existing_child_tuple[0], Atom::str_to_term("ok"));
    let existing_child = ok_existing_child_tuple[1];
    assert!(existing_child.is_boxed_resource_reference());

    assert!(document.is_boxed_resource_reference());

    process.queue_frame_with_arguments(
        label_3::frame().with_arguments(true, &[document, existing_child]),
    );

    let parent_tag = process.binary_from_str("div")?;
    process.queue_frame_with_arguments(
        liblumen_web::document::create_element_2::frame()
            .with_arguments(false, &[document, parent_tag]),
    );

    Ok(Term::NONE)
}
