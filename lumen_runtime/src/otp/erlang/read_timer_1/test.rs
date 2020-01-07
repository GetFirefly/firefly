mod with_local_reference;

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::otp::erlang;
use crate::otp::erlang::read_timer_1::native;
use crate::process;
use crate::test::{has_message, receive_message, timeout_message};
use crate::time::Milliseconds;
use crate::timer;

#[test]
fn without_reference_errors_badarg() {
    crate::test::without_timer_reference_errors_badarg(file!(), native);
}
