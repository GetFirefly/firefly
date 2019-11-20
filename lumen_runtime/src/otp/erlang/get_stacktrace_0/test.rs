mod with_exception;

use liblumen_alloc::erts::term::prelude::Term;

use crate::otp::erlang::get_stacktrace_0::native;
use crate::scheduler::with_process;

#[test]
fn without_exception_returns_empty_list() {
    with_process(|process| {
        assert_eq!(native(process), Ok(Term::NIL));
    });
}
