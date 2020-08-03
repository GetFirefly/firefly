mod with_big_integer_dividend;
mod with_small_integer_dividend;

use proptest::prop_assert_eq;
use proptest::strategy::{BoxedStrategy, Just};

use liblumen_alloc::erts::term::prelude::*;

use crate::erlang;
use crate::erlang::div_2::result;
use crate::test::strategy;
use crate::test::with_process;

#[test]
fn without_integer_dividend_errors_badarith() {
    crate::test::without_integer_dividend_errors_badarith(file!(), result);
}

#[test]
fn with_integer_dividend_without_integer_divisor_errors_badarith() {
    crate::test::with_integer_dividend_without_integer_divisor_errors_badarith(file!(), result);
}

#[test]
fn with_integer_dividend_with_zero_divisor_errors_badarith() {
    crate::test::with_integer_dividend_with_zero_divisor_errors_badarith(file!(), result);
}
