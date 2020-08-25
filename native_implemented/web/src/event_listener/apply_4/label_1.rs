use liblumen_alloc::erts::term::prelude::*;

#[native_implemented::label]
fn result(
    apply_returned: Term,
    // Having event_listener as an argument ensures it is not dropped while the apply/3 is running.
    _event_listener: Term,
) -> Term {
    apply_returned
}
