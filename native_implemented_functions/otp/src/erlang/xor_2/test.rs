mod with_false_left;
mod with_true_left;

use crate::erlang::xor_2::result;
use crate::test::strategy;

#[test]
fn without_boolean_left_errors_badarg() {
    crate::test::without_boolean_left_errors_badarg(file!(), result);
}
