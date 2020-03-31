mod with_function;

use std::convert::TryInto;
use std::sync::Arc;

use anyhow::*;

use proptest::strategy::Strategy;
use proptest::test_runner::{Config, TestRunner};
use proptest::{prop_assert, prop_assert_eq};

use liblumen_alloc::erts::process::{Process, Status};
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::{atom, exit};

use crate::runtime::registry::pid_to_process;

use crate::runtime::scheduler;

use crate::erlang::spawn_monitor_1::native;
use crate::test;
use crate::test::strategy;
use crate::test::strategy::term::function;
use crate::test::{badarity_reason, has_message, prop_assert_exits_badarity};

#[test]
fn without_function_errors_badarg() {
    test::without_function_errors_badarg(file!(), native);
}
