use crate::erlang::spawn_link_1::result;
use crate::test;

#[test]
fn without_function_errors_badarg() {
    test::without_function_errors_badarg(file!(), result);
}
