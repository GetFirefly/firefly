//! ```elixir
//! # label 5
//! # pushed to stack: (child)
//! # returned from call: :ok
//! # full stack: (:ok, child)
//! # returns: :ok
//! remove_ok = Lumen.Web.Element.remove(child);
//! Lumen.Web.Wait.with_return(remove_ok)
//! ```

use liblumen_alloc::erts::process::{Frame, Native, Process};
use liblumen_alloc::erts::term::prelude::*;

use liblumen_web::runtime::process::current_process;

pub fn frame() -> Frame {
    super::frame(NATIVE)
}

// Private

const NATIVE: Native = Native::Two(native);

extern "C" fn native(ok: Term, child: Term) -> Term {
    let arc_process = current_process();
    arc_process.reduce();

    result(&arc_process, ok, child)
}

fn result(process: &Process, ok: Term, child: Term) -> Term {
    assert_eq!(ok, Atom::str_to_term("ok"));
    assert!(child.is_boxed_resource_reference());

    process.queue_frame_with_arguments(
        liblumen_web::element::remove_1::frame().with_arguments(false, &[child]),
    );

    Term::NONE
}
