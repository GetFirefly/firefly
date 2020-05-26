//! ```elixir
//! # label 6
//! # pushed to stack: (parent, reference_child)
//! # returned form call: {:ok, new_child}
//! # full stack: ({:ok, new_child}, parent, reference_child)
//! # returns: {:ok, inserted_child}
//! {:ok, inserted_child} = Lumen.Web.insert_before(parent, new_child, reference_child)
//! ```

use std::convert::TryInto;

use liblumen_alloc::erts::process::{Frame, Native, Process};
use liblumen_alloc::erts::term::prelude::*;

use liblumen_web::runtime::process::current_process;

pub fn frame() -> Frame {
    super::frame(NATIVE)
}

// Private

const NATIVE: Native = Native::Three(native);

extern "C" fn native(ok_old_child: Term, parent: Term, document: Term) -> Term {
    let arc_process = current_process();
    arc_process.reduce();

    result(&arc_process, ok_old_child, parent, document)
}

fn result(process: &Process, ok_new_child: Term, parent: Term, reference_child: Term) -> Term {
    assert!(
        ok_new_child.is_boxed_tuple(),
        "ok_new_child ({:?}) is not a tuple",
        ok_new_child
    );
    let ok_new_child_tuple: Boxed<Tuple> = ok_new_child.try_into().unwrap();
    assert_eq!(ok_new_child_tuple.len(), 2);
    assert_eq!(ok_new_child_tuple[0], Atom::str_to_term("ok"));
    let new_child = ok_new_child_tuple[1];
    assert!(new_child.is_boxed_resource_reference());

    assert!(parent.is_boxed_resource_reference());
    assert!(reference_child.is_boxed_resource_reference());

    process.queue_frame_with_arguments(
        liblumen_web::node::insert_before_3::frame()
            .with_arguments(false, &[parent, new_child, reference_child]),
    );

    Term::NONE
}
