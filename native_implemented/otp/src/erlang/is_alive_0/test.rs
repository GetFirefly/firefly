use crate::erlang::is_alive_0::result;

#[test]
fn returns_false() {
    assert_eq!(result(), false.into())
}
