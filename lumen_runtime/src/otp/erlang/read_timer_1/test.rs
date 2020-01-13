mod with_local_reference;

use liblumen_alloc::erts::term::prelude::*;

use crate::otp::erlang::read_timer_1::native;

#[test]
fn without_reference_errors_badarg() {
    crate::test::without_timer_reference_errors_badarg(file!(), native);
}
