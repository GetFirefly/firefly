mod with_integer_integer;

use proptest::strategy::Just;
use proptest::{prop_assert, prop_assert_eq};

use liblumen_alloc::erts::term::prelude::*;

use crate::erlang::integer_to_binary_2::result;
use crate::test::strategy;

#[test]
fn without_integer_integer_errors_badarg() {
    crate::test::without_integer_integer_with_base_errors_badarg(file!(), result);
}
