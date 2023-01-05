mod with_exception;

use firefly_rt::term::{atoms, Term};

use crate::erlang::get_stacktrace_0::result;
use crate::test::with_process;

#[test]
fn without_exception_returns_empty_list() {
    with_process(|process| {
        assert_eq!(result(process), Term::Nil);
    });
}
