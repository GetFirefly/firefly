mod with_float;

use proptest::strategy::Just;
use proptest::test_runner::{Config, TestRunner};
use proptest::{prop_assert, prop_assert_eq};

use firefly_rt::process::Process;
use firefly_rt::term::{atoms, Term};

use crate::erlang::float_to_binary_2::result;
use crate::test::with_process_arc;
use crate::test::{strategy, without_float_with_empty_options_errors_badarg};

#[test]
fn without_float_errors_badarg() {
    without_float_with_empty_options_errors_badarg(file!(), result);
}
