//! ```elixir
//! # label 5
//! # pushed to stack: (parent, old_child)
//! # returned form call: {:ok, new_child}
//! # full stack: ({:ok, new_child}, parent, old_child)
//! # returns: {:ok, replaced_child}
//! {:ok, replaced_child} = Lumen.Web.replace_child(parent, new_child, old_child)
//! ```

use std::convert::TryInto;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

#[native_implemented::label]
fn result(
    process: &Process,
    ok_new_child: Term,
    parent: Term,
    old_child: Term,
) -> exception::Result<Term> {
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
    assert!(old_child.is_boxed_resource_reference());

    process.queue_frame_with_arguments(
        liblumen_web::node::replace_child_3::frame()
            .with_arguments(false, &[parent, new_child, old_child]),
    );

    Ok(Term::NONE)
}
