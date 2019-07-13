use super::*;

#[test]
fn returns_nonode_at_nohost() {
    assert_eq!(erlang::node_0(), atom_unchecked("nonode@nohost"))
}
