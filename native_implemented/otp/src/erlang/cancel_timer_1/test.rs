mod with_local_reference;

use liblumen_alloc::erts::term::prelude::*;

use crate::erlang::cancel_timer_1::result;
use crate::test::{
    freeze_at_timeout, freeze_timeout, has_message, timeout_message, with_timer_in_same_thread,
};

#[test]
fn without_reference_errors_badarg() {
    crate::test::without_timer_reference_errors_badarg(file!(), result);
}
