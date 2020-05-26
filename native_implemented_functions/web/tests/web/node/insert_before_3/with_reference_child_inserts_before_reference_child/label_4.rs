//! ```elixir
//! # label 4
//! # pushed to stack: (document, parent, reference_child)
//! # returned form call: :ok
//! # full stack: (:ok, document, parent, reference_child)
//! # returns: :ok
//! :ok = Lumen.Web.Node.append_child(parent, reference_child)
//! {:ok, new_child} = Lumen.Web.Document.create_element(document, "ul");
//! {:ok, inserted_child} = Lumen.Web.insert_before(parent, new_child, reference_child)
//! ```

use liblumen_alloc::erts::process::{Frame, Native, Process};
use liblumen_alloc::erts::term::prelude::*;

use liblumen_web::runtime::process::current_process;

use super::label_5;

pub fn frame() -> Frame {
    super::frame(NATIVE)
}

// Private

const NATIVE: Native = Native::Four(native);

extern "C" fn native(ok: Term, document: Term, parent: Term, reference_child: Term) -> Term {
    let arc_process = current_process();
    arc_process.reduce();

    result(&arc_process, ok, document, parent, reference_child)
}

fn result(
    process: &Process,
    ok: Term,
    document: Term,
    parent: Term,
    reference_child: Term,
) -> Term {
    assert_eq!(ok, Atom::str_to_term("ok"));
    assert!(document.is_boxed_resource_reference());
    assert!(parent.is_boxed_resource_reference());
    assert!(reference_child.is_boxed_resource_reference());

    process.queue_frame_with_arguments(
        label_5::frame().with_arguments(true, &[document, parent, reference_child]),
    );

    process.queue_frame_with_arguments(
        liblumen_web::node::append_child_2::frame()
            .with_arguments(false, &[parent, reference_child]),
    );

    Term::NONE
}
