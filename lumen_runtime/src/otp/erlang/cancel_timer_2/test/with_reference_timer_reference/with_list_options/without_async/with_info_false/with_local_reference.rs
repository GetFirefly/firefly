use super::*;

mod with_timer;

#[test]
fn without_timer_returns_ok() {
    with_info_false_without_timer_returns_ok(options);
}
