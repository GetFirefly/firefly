use crate::otp::erlang::is_alive_0::native;

#[test]
fn returns_false() {
    assert_eq!(native(), false.into())
}
