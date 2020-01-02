mod with_false_left;
mod with_true_left;

use proptest::test_runner::{Config, TestRunner};

use crate::otp::erlang::xor_2::native;
use crate::scheduler::with_process_arc;
use crate::test::strategy;

#[test]
fn without_boolean_left_errors_badarg() {
    crate::test::without_boolean_left_errors_badarg(file!(), native);
}
