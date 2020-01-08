mod with_big_integer_dividend;
mod with_small_integer_dividend;

use proptest::prop_assert_eq;
use proptest::strategy::{BoxedStrategy, Just};

use liblumen_alloc::erts::term::prelude::*;

use crate::otp::erlang;
use crate::otp::erlang::div_2::native;
use crate::scheduler::with_process;
use crate::test::strategy;

#[test]
fn without_integer_dividend_errors_badarith() {
    crate::test::without_integer_dividend_errors_badarith(file!(), native);
}

#[test]
fn with_integer_dividend_without_integer_divisor_errors_badarith() {
    crate::test::with_integer_dividend_without_integer_divisor_errors_badarith(file!(), native);
}

#[test]
fn with_integer_dividend_with_zero_divisor_errors_badarith() {
    crate::test::with_integer_dividend_with_zero_divisor_errors_badarith(file!(), native);
}
