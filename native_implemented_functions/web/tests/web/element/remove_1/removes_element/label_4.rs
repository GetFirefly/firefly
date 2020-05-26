//! ```elixir
//! # label 3
//! # pushed to stack: (body)
//! # returned from call: {:ok, child}
//! # full stack: ({:ok, child}, body)
//! # returns: :ok
//! :ok = Lumen.Web.Node.append_child(body, child);
//! remove_ok = Lumen.Web.Element.remove(child);
//! Lumen.Web.Wait.with_return(remove_ok)
//! ```

use std::convert::TryInto;

use web_sys::Element;

use liblumen_alloc::erts::process::{Frame, Native, Process};
use liblumen_alloc::erts::term::prelude::*;

use liblumen_web::runtime::process::current_process;

use super::label_5;

pub fn frame() -> Frame {
    super::frame(NATIVE)
}

// Private

const NATIVE: Native = Native::Two(native);

extern "C" fn native(ok_child: Term, body: Term) -> Term {
    let arc_process = current_process();
    arc_process.reduce();

    result(&arc_process, ok_child, body)
}

fn result(process: &Process, ok_child: Term, body: Term) -> Term {
    assert!(
        ok_child.is_boxed_tuple(),
        "ok_child ({:?}) is not a tuple",
        ok_child
    );
    let ok_child_tuple: Boxed<Tuple> = ok_child.try_into().unwrap();
    assert_eq!(ok_child_tuple.len(), 2);
    assert_eq!(ok_child_tuple[0], Atom::str_to_term("ok"));
    let child = ok_child_tuple[1];
    let child_ref_boxed: Boxed<Resource> = child.try_into().unwrap();
    let child_reference: Resource = child_ref_boxed.into();
    let _: &Element = child_reference.downcast_ref().unwrap();

    assert!(body.is_boxed_resource_reference());

    process.queue_frame_with_arguments(label_5::frame().with_arguments(true, &[child]));

    process.queue_frame_with_arguments(
        liblumen_web::node::append_child_2::frame().with_arguments(false, &[body, child]),
    );

    Term::NONE
}
