mod with_function;

use std::convert::TryInto;

use anyhow::*;

use proptest::prop_assert_eq;
use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::atom;
use liblumen_alloc::erts::process::Status;

use crate::otp::erlang::spawn_monitor_1::native;
use crate::registry::pid_to_process;
use crate::test::strategy;
use crate::test::strategy::term::function;

#[test]
fn without_function_errors_badarg() {
    crate::test::without_function_errors_badarg(file!(), native);
}
