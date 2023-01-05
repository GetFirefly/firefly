mod with_integer_integer;

use proptest::{prop_assert, prop_assert_eq};
use proptest::arbitrary::any;
use proptest::strategy::Just;

use firefly_rt::term::Term;

use crate::erlang::integer_to_list_2::result;
use crate::erlang::list_to_string::list_to_string;
use crate::test::strategy;

#[test]
fn without_integer_integer_errors_badarg() {
    crate::test::without_integer_integer_with_base_errors_badarg(file!(), result);
}
