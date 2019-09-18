use liblumen_alloc::erts::term::atom_unchecked;

use crate::otp::erlang::node_0::native;

#[test]
fn returns_nonode_at_nohost() {
    assert_eq!(native(), atom_unchecked("nonode@nohost"))
}
