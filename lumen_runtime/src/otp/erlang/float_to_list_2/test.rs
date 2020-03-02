mod with_float;

use proptest::strategy::Just;
use proptest::test_runner::{Config, TestRunner};
use proptest::{prop_assert, prop_assert_eq};

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::otp::erlang::float_to_list_2::native;
use crate::scheduler::with_process_arc;
use crate::test::strategy;

#[test]
fn without_float_errors_badarg() {
    crate::test::without_float_with_empty_options_errors_badarg(file!(), native);
}
