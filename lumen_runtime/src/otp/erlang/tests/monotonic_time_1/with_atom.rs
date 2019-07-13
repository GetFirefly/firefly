use super::*;

use std::thread;
use std::time::Duration;

#[test]
fn with_invalid_unit_errors_badarg() {
    errors_badarg(|_| atom_unchecked("invalid"));
}

#[test]
fn with_second_increases_after_2_seconds() {
    with_process(|process| {
        let unit = atom_unchecked("second");

        let first = erlang::monotonic_time_1(unit, process).unwrap();

        thread::sleep(Duration::from_secs(2));

        let second = erlang::monotonic_time_1(unit, process).unwrap();

        assert_eq!(erlang::is_less_than_2(first, second), true.into());
    });
}

#[test]
fn with_millisecond_increases_after_2_milliseconds() {
    with_process(|process| {
        let unit = atom_unchecked("millisecond");

        let first = erlang::monotonic_time_1(unit, process).unwrap();

        thread::sleep(Duration::from_millis(2));

        let second = erlang::monotonic_time_1(unit, process).unwrap();

        assert_eq!(erlang::is_less_than_2(first, second), true.into());
    });
}

#[test]
fn with_microsecond_increases_after_2_milliseconds() {
    with_process(|process| {
        let unit = atom_unchecked("microsecond");

        let first = erlang::monotonic_time_1(unit, process).unwrap();

        thread::sleep(Duration::from_millis(2));

        let second = erlang::monotonic_time_1(unit, process).unwrap();

        assert_eq!(erlang::is_less_than_2(first, second), true.into());
    });
}

#[test]
fn with_nanosecond_increases_after_2_milliseconds() {
    with_process(|process| {
        let unit = atom_unchecked("nanosecond");

        let first = erlang::monotonic_time_1(unit, process).unwrap();

        thread::sleep(Duration::from_millis(2));

        let second = erlang::monotonic_time_1(unit, process).unwrap();

        assert_eq!(erlang::is_less_than_2(first, second), true.into());
    });
}

#[test]
fn with_native_increases_after_2_native_time_units() {
    with_process(|process| {
        let unit = atom_unchecked("native");

        let first = erlang::monotonic_time_1(unit, process).unwrap();

        thread::sleep(Duration::from_millis(2));

        let second = erlang::monotonic_time_1(unit, process).unwrap();

        assert_eq!(erlang::is_less_than_2(first, second), true.into());
    });
}

#[test]
fn with_perf_counter_increases_after_2_perf_counter_ticks() {
    with_process(|process| {
        let unit = atom_unchecked("perf_counter");

        let first = erlang::monotonic_time_1(unit, process).unwrap();

        thread::sleep(Duration::from_millis(2));

        let second = erlang::monotonic_time_1(unit, process).unwrap();

        assert_eq!(erlang::is_less_than_2(first, second), true.into());
    });
}
