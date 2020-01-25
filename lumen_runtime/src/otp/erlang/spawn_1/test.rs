mod with_function;

use std::convert::TryInto;

use proptest::prop_assert_eq;
use proptest::strategy::Just;

use liblumen_alloc::erts::process::Status;
use liblumen_alloc::erts::term::closure::Definition;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::located_code;

use crate::otp::erlang::spawn_1::native;
use crate::registry::pid_to_process;
use crate::test::strategy::term::function;
use crate::test::{prop_assert_exits_badarity, strategy};

#[test]
fn without_function_errors_badarg() {
    crate::test::without_function_errors_badarg(file!(), native);
}
