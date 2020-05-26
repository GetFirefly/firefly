//! ```elixir
//! # label 3
//! # pushed to stack: ()
//! # returned from call: {:ok, body}
//! # full stack: ({:ok, body})
//! # returns: class_name
//! class_name = Lumen.Web.Element.class_name(body)
//! Lumen.Web.Wait.with_return(class_name)
//! ```

use std::convert::TryInto;

use liblumen_alloc::erts::process::{Frame, Native, Process};
use liblumen_alloc::erts::term::prelude::*;

use liblumen_web::runtime::process::current_process;

pub fn frame() -> Frame {
    super::frame(NATIVE)
}

// Private

const NATIVE: Native = Native::One(native);

extern "C" fn native(ok_body: Term) -> Term {
    let arc_process = current_process();
    arc_process.reduce();

    result(&arc_process, ok_body)
}

fn result(process: &Process, ok_body: Term) -> Term {
    assert!(
        ok_body.is_boxed_tuple(),
        "ok_body ({:?}) is not a tuple",
        ok_body
    );
    let ok_body_tuple: Boxed<Tuple> = ok_body.try_into().unwrap();
    assert_eq!(ok_body_tuple.len(), 2);
    assert_eq!(ok_body_tuple[0], Atom::str_to_term("ok"));
    let body = ok_body_tuple[1];
    assert!(body.is_boxed_resource_reference());

    process.queue_frame_with_arguments( liblumen_web::element::class_name_1::frame().with_arguments(false, &[body]));

    Term::NONE
}
