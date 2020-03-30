mod with_entries;

use liblumen_alloc::erts::term::prelude::Term;

use crate::erlang::erase_0::native;
use crate::test::with_process;

#[test]
fn without_entries_returns_empty_list() {
    with_process(|process| {
        assert_eq!(native(process), Ok(Term::NIL));
    });
}
