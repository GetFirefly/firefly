mod with_binary;

use proptest::strategy::{Just, Strategy};
use proptest::{prop_assert, prop_assert_eq};

use crate::erlang::binary_to_integer_1::result;
use crate::test::strategy;

#[test]
fn without_binary_errors_badarg() {
    crate::test::without_binary_errors_badarg(file!(), result);
}
