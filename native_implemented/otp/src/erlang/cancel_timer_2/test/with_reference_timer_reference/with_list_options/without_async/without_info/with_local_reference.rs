use super::*;

mod with_timer;

#[test]
fn without_timer_returns_false() {
    returns_false(options);
}
