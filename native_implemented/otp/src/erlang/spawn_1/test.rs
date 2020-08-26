// `with_function` in integration tests

use crate::erlang::spawn_1::result;

#[test]
fn without_function_errors_badarg() {
    crate::test::without_function_errors_badarg(file!(), result);
}
