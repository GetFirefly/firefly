//! ```elixir
//! # label 4
//! # pushed to stack: (document, parent, old_child)
//! # returned form call: :ok
//! # full stack: (:ok, document, parent, old_child)
//! # returns: {:ok, new_child}
//! {:ok, new_child} = Lumen.Web.Document.create_element(document, "ul");
//! {:ok, replaced_child} = Lumen.Web.replace_child(parent, new_child, old_child)
//! ```

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use super::label_5;

#[native_implemented::label]
fn result(
    process: &Process,
    ok: Term,
    document: Term,
    parent: Term,
    old_child: Term,
) -> exception::Result<Term> {
    assert_eq!(ok, Atom::str_to_term("ok"));
    assert!(document.is_boxed_resource_reference());
    assert!(parent.is_boxed_resource_reference());
    assert!(old_child.is_boxed_resource_reference());

    let new_child_tag = process.binary_from_str("ul");
    process.queue_frame_with_arguments(
        liblumen_web::document::create_element_2::frame()
            .with_arguments(false, &[document, new_child_tag]),
    );

    process.queue_frame_with_arguments(label_5::frame().with_arguments(true, &[parent, old_child]));

    Ok(Term::NONE)
}
