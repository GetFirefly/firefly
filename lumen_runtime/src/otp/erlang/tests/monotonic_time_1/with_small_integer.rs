use super::*;

use std::thread;
use std::time::Duration;

#[test]
fn with_negative_errors_badarg() {
    errors_badarg(|process| (-1).into_process(process));
}

#[test]
fn with_zero_errors_badarg() {
    errors_badarg(|process| 0.into_process(process));
}

#[test]
fn with_positive_increases_after_2_time_units() {
    with_process(|process| {
        let unit = 2.into_process(process);

        let first = erlang::monotonic_time_1(unit, process).unwrap();

        thread::sleep(Duration::from_secs(1));

        let second = erlang::monotonic_time_1(unit, process).unwrap();

        assert_eq!(erlang::is_less_than_2(first, second), true.into());
    });
}
