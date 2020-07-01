mod with_float;

use proptest::test_runner::{Config, TestRunner};
use proptest::{prop_assert, prop_assert_eq};

use liblumen_alloc::erts::term::prelude::*;

use crate::erlang::float_to_list_1::result;
use crate::test::strategy;
use crate::test::with_process_arc;

#[test]
fn without_float_errors_badarg() {
    crate::test::without_float_errors_badarg(file!(), result);
}
