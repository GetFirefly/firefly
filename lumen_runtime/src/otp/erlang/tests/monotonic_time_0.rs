use super::*;

use std::thread;
use std::time::Duration;

#[test]
fn increases_after_2_native_time_units() {
    with_process(|process| {
        let first = erlang::monotonic_time_0(process);

        thread::sleep(Duration::from_millis(2));

        let second = erlang::monotonic_time_0(process);

        assert_eq!(erlang::is_less_than_2(first, second), true.into())
    });
}
