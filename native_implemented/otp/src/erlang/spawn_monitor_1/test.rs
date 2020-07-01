mod with_function;

use std::convert::TryInto;

use anyhow::*;

use proptest::strategy::Strategy;
use proptest::test_runner::{Config, TestRunner};
use proptest::{prop_assert, prop_assert_eq};

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::{Process, Status};
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::{atom, exit};

use crate::erlang::spawn_monitor_1::result;
use crate::runtime::process::current_process;
use crate::runtime::registry::pid_to_process;
use crate::runtime::scheduler;
use crate::test;
use crate::test::strategy::term::function;
use crate::test::*;

#[test]
fn without_function_errors_badarg() {
    test::without_function_errors_badarg(file!(), result);
}
