//! ```elixir
//! # label 12
//! # pushed to stack: (tr)
//! # returned from call: {:ok, tbody}
//! # full stack: ({:ok, tbody}, tr)
//! # returns: :ok
//! Lumen::Web::Node.append_child(tbody, tr)
//! ```

use std::convert::TryInto;

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

// Private

#[native_implemented::label]
fn result(process: &Process, ok_tbody: Term, tr: Term) -> Term {
    assert!(
        ok_tbody.is_boxed_tuple(),
        "ok_tbody ({:?}) is not a tuple",
        ok_tbody
    );
    assert!(tr.is_boxed_resource_reference());

    let ok_tbody_tuple: Boxed<Tuple> = ok_tbody.try_into().unwrap();
    assert_eq!(ok_tbody_tuple.len(), 2);
    assert_eq!(ok_tbody_tuple[0], Atom::str_to_term("ok"));
    let tbody = ok_tbody_tuple[1];
    assert!(tbody.is_boxed_resource_reference());

    process.queue_frame_with_arguments(
        liblumen_web::node::append_child_2::frame().with_arguments(false, &[tbody, tr]),
    );

    Term::NONE
}
