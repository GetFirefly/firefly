mod with_local_reference;

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::otp::erlang;
use crate::otp::erlang::cancel_timer_1::native;
use crate::test::{has_message, timeout_message};
use crate::{process, timer};

#[test]
fn without_reference_errors_badarg() {
    crate::test::without_timer_reference_errors_badarg(file!(), native);
}
