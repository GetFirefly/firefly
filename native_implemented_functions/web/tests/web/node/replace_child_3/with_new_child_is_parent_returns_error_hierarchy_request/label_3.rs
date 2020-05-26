//! ```elixir
//! # label 3
//! # pushed to stack: (old_child)
//! # returned form call: {:ok, parent}
//! # full stack: ({:ok, parent}, old_child)
//! # returns: :ok
//! :ok = Lumen.Web.Node.append_child(parent, old_child)
//! {:error, :hierarchy_request} = Lumen.Web.replace_child(parent, old_child, parent)
//! ```

use std::convert::TryInto;

use liblumen_alloc::erts::process::{Frame, Native, Process};
use liblumen_alloc::erts::term::prelude::*;

use liblumen_web::runtime::process::current_process;

use super::label_4;

pub fn frame() -> Frame {
    super::frame(NATIVE)
}

// Private

const NATIVE: Native = Native::Two(native);

extern "C" fn native(ok_parent: Term, old_child: Term) -> Term {
    let arc_process = current_process();
    arc_process.reduce();

    result(&arc_process, ok_parent, old_child)
}

fn result(process: &Process, ok_parent: Term, old_child: Term) -> Term {
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

    assert!(old_child.is_boxed_resource_reference());

    process.queue_frame_with_arguments(label_4::frame().with_arguments(true, &[parent, old_child]));
    process.queue_frame_with_arguments(
        liblumen_web::node::append_child_2::frame().with_arguments(false, &[parent, old_child]),
    );

    Term::NONE
}
