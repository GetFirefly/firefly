use super::*;

use std::thread;
use std::time::Duration;

#[test]
fn with_invalid_unit_errors_badarg() {
    errors_badarg(|_| Term::str_to_atom("invalid", DoNotCare).unwrap());
}

#[test]
fn with_second_increases_after_2_seconds() {
    with_process(|process| {
        let unit = Term::str_to_atom("second", DoNotCare).unwrap();

        let first = erlang::monotonic_time_1(unit, process).unwrap();

        thread::sleep(Duration::from_secs(2));

        let second = erlang::monotonic_time_1(unit, process).unwrap();

        assert_eq!(erlang::is_less_than_2(first, second), true.into());
    });
}

#[test]
fn with_millisecond_increases_after_2_milliseconds() {
    with_process(|process| {
        let unit = Term::str_to_atom("millisecond", DoNotCare).unwrap();

        let first = erlang::monotonic_time_1(unit, process).unwrap();

        thread::sleep(Duration::from_millis(2));

        let second = erlang::monotonic_time_1(unit, process).unwrap();

        assert_eq!(erlang::is_less_than_2(first, second), true.into());
    });
}

#[test]
fn with_microsecond_increases_after_2_microseconds() {
    with_process(|process| {
        let unit = Term::str_to_atom("microsecond", DoNotCare).unwrap();

        let first = erlang::monotonic_time_1(unit, process).unwrap();

        thread::sleep(Duration::from_micros(2));

        let second = erlang::monotonic_time_1(unit, process).unwrap();

        assert_eq!(erlang::is_less_than_2(first, second), true.into());
    });
}

#[test]
fn with_nanosecond_increases_after_2_nanoseconds() {
    with_process(|process| {
        let unit = Term::str_to_atom("nanosecond", DoNotCare).unwrap();

        let first = erlang::monotonic_time_1(unit, process).unwrap();

        thread::sleep(Duration::from_nanos(2));

        let second = erlang::monotonic_time_1(unit, process).unwrap();

        assert_eq!(erlang::is_less_than_2(first, second), true.into());
    });
}

#[test]
fn with_native_increases_after_2_native_time_units() {
    with_process(|process| {
        let unit = Term::str_to_atom("native", DoNotCare).unwrap();

        let first = erlang::monotonic_time_1(unit, process).unwrap();

        thread::sleep(Duration::from_millis(2));

        let second = erlang::monotonic_time_1(unit, process).unwrap();

        assert_eq!(erlang::is_less_than_2(first, second), true.into());
    });
}

#[test]
fn with_perf_counter_increases_after_2_perf_counter_ticks() {
    with_process(|process| {
        let unit = Term::str_to_atom("perf_counter", DoNotCare).unwrap();

        let first = erlang::monotonic_time_1(unit, process).unwrap();

        thread::sleep(Duration::from_millis(2));

        let second = erlang::monotonic_time_1(unit, process).unwrap();

        assert_eq!(erlang::is_less_than_2(first, second), true.into());
    });
}
