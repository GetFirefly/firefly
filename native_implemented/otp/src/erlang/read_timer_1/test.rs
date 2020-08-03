mod with_local_reference;

use liblumen_alloc::erts::term::prelude::*;

use crate::erlang::read_timer_1::result;

#[test]
fn without_reference_errors_badarg() {
    crate::test::without_timer_reference_errors_badarg(file!(), result);
}
