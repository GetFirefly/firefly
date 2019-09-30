mod with_entries;

use liblumen_alloc::erts::term::Term;

use crate::otp::erlang::get_0::native;
use crate::scheduler::with_process;

#[test]
fn without_entries_returns_empty_list() {
    with_process(|process| {
        assert_eq!(native(process), Ok(Term::NIL));
    });
}
