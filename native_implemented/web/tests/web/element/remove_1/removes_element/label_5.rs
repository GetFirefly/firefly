//! ```elixir
//! # label 5
//! # pushed to stack: (child)
//! # returned from call: :ok
//! # full stack: (:ok, child)
//! # returns: :ok
//! remove_ok = Lumen.Web.Element.remove(child);
//! Lumen.Web.Wait.with_return(remove_ok)
//! ```

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

#[native_implemented::label]
fn result(process: &Process, ok: Term, child: Term) -> Term {
    assert_eq!(ok, Atom::str_to_term("ok"));
    assert!(child.is_boxed_resource_reference());

    process.queue_frame_with_arguments(
        liblumen_web::element::remove_1::frame().with_arguments(false, &[child]),
    );

    Term::NONE
}
