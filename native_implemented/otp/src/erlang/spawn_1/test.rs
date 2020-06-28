mod with_function;

use std::convert::TryInto;

use proptest::prop_assert_eq;
use proptest::strategy::Just;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Status;
use liblumen_alloc::erts::term::prelude::*;

use crate::erlang::spawn_1::result;
use crate::runtime::process::current_process;
use crate::runtime::registry::pid_to_process;
use crate::test::strategy::term::function;
use crate::test::*;

#[test]
fn without_function_errors_badarg() {
    crate::test::without_function_errors_badarg(file!(), result);
}
