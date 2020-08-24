// `with_function` in integration tests

use crate::erlang::spawn_monitor_1::result;
use crate::test;

#[test]
fn without_function_errors_badarg() {
    test::without_function_errors_badarg(file!(), result);
}
