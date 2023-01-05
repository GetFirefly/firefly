//! ```elixir
//! # label 3
//! # pushed to stack: (document, old_child)
//! # returned form call: {:ok, parent}
//! # full stack: ({:ok, parent}, document, old_child)
//! # returns: :ok
//! :ok = Lumen.Web.Node.append_child(parent, old_child)
//! {:ok, new_child} = Lumen.Web.Document.create_element(document, "ul");
//! {:ok, replaced_child} = Lumen.Web.replace_child(parent, new_child, old_child)
//! ```

use std::convert::TryInto;

use liblumen_rt::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use super::label_4;

#[native_implemented::label]
fn result(process: &Process, ok_parent: Term, document: Term, old_child: Term) -> Term {
    assert!(
        ok_parent.is_boxed_tuple(),
        "ok_parent ({:?}) is not a tuple",
        ok_parent
    );
    let ok_parent_tuple: Boxed<Tuple> = ok_parent.try_into().unwrap();
    assert_eq!(ok_parent_tuple.len(), 2);
    assert_eq!(ok_parent_tuple[0], Atom::str_to_term("ok"));
    let parent = ok_parent_tuple[1];
    assert!(parent.is_boxed_resource_reference());

    assert!(document.is_boxed_resource_reference());
    assert!(old_child.is_boxed_resource_reference());

    process.queue_frame_with_arguments(
        liblumen_web::node::append_child_2::frame().with_arguments(false, &[parent, old_child]),
    );
    process.queue_frame_with_arguments(
        label_4::frame().with_arguments(true, &[document, parent, old_child]),
    );

    Term::NONE
}
