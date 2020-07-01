//! ```elixir
//! # label 2
//! # pushed to stack: (parent_document)
//! # returned form call: {:ok, old_child}
//! # full stack: ({:ok, old_child}, parent_document)
//! # returns: {:ok, parent}
//! {:ok, parent} = Lumen.Web.Document.create_element(parent_document, "div")
//! :ok = Lumen.Web.Node.append_child(parent, old_child)
//! {:ok, new_child} = Lumen.Web.Document.create_element(document, "ul");
//! {:ok, replaced_child} = Lumen.Web.replace_child(parent, new_child, old_child)
//! ```

use std::convert::TryInto;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use super::label_3;

#[native_implemented::label]
fn result(process: &Process, ok_old_child: Term, document: Term) -> exception::Result<Term> {
    assert!(
        ok_old_child.is_boxed_tuple(),
        "ok_old_child ({:?}) is not a tuple",
        ok_old_child
    );
    let ok_old_child_tuple: Boxed<Tuple> = ok_old_child.try_into().unwrap();
    assert_eq!(ok_old_child_tuple.len(), 2);
    assert_eq!(ok_old_child_tuple[0], Atom::str_to_term("ok"));
    let old_child = ok_old_child_tuple[1];
    assert!(old_child.is_boxed_resource_reference());
    assert!(document.is_boxed_resource_reference());

    let parent_tag = process.binary_from_str("div")?;
    process.queue_frame_with_arguments(
        liblumen_web::document::create_element_2::frame()
            .with_arguments(false, &[document, parent_tag]),
    );

    process
        .queue_frame_with_arguments(label_3::frame().with_arguments(true, &[document, old_child]));

    Ok(Term::NONE)
}
