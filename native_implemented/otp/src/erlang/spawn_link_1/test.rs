mod with_function;

use std::convert::TryInto;

use anyhow::*;

use proptest::strategy::Strategy;
use proptest::test_runner::{Config, TestRunner};
use proptest::{prop_assert, prop_assert_eq};

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::{Process, Status};
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::exit;

use crate::erlang::spawn_link_1::result;
use crate::runtime::process::current_process;
use crate::runtime::registry::pid_to_process;
use crate::runtime::scheduler;
use crate::test;
use crate::test::prop_assert_exits_badarity;
use crate::test::strategy;
use crate::test::strategy::term::function;

#[test]
fn without_function_errors_badarg() {
    test::without_function_errors_badarg(file!(), result);
}
