mod with_integer;

use proptest::strategy::Just;
use proptest::{prop_assert, prop_assert_eq};

use liblumen_alloc::erts::term::prelude::*;

use crate::erlang::integer_to_binary_1::result;
use crate::test::strategy;

#[test]
fn without_integer_errors_badarg() {
    crate::test::without_integer_errors_badarg(file!(), result);
}
