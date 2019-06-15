use super::*;

#[test]
fn returns_false() {
    assert_eq!(erlang::is_alive_0(), false.into())
}
