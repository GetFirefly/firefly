// `with_integer` in integration tests

use crate::erlang::integer_to_binary_1::result;

#[test]
fn without_integer_errors_badarg() {
    crate::test::without_integer_errors_badarg(file!(), result);
}
