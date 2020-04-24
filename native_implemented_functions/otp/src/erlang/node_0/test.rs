use liblumen_alloc::erts::term::prelude::Atom;

use crate::erlang::node_0::result;

#[test]
fn returns_nonode_at_nohost() {
    assert_eq!(result(), Atom::str_to_term("nonode@nohost"))
}
