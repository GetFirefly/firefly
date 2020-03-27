mod with_function;

use std::convert::TryInto;
use std::sync::Arc;

use anyhow::*;

use proptest::strategy::Strategy;
use proptest::test_runner::{Config, TestRunner};
use proptest::{prop_assert, prop_assert_eq};

use liblumen_alloc::erts::process::{Process, Status};
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::exit;

use lumen_runtime::registry::pid_to_process;
use lumen_runtime::scheduler::Scheduler;

use crate::erlang::spawn_link_1::native;
use crate::test;
use crate::test::prop_assert_exits_badarity;
use crate::test::strategy;
use crate::test::strategy::term::function;

#[test]
fn without_function_errors_badarg() {
    test::without_function_errors_badarg(file!(), native);
}
