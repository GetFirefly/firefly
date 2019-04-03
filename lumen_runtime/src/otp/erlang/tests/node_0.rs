use super::*;

#[test]
fn returns_nonode_at_nohost() {
    assert_eq!(
        erlang::node_0(),
        Term::str_to_atom("nonode@nohost", DoNotCare).unwrap()
    )
}
