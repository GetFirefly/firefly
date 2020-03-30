use super::*;

use std::thread;
use std::time::Duration;

#[test]
fn with_invalid_unit_errors_badarg() {
    with_process(|process| {
        assert_badarg!(
            native(process, atom!("invalid")),
            "atom (invalid) is not supported"
        );
    });
}

#[test]
fn with_second_increases_after_2_seconds() {
    with_process(|process| {
        let unit = Atom::str_to_term("second");

        let first = native(process, unit).unwrap();

        thread::sleep(Duration::from_secs(2));

        let second = native(process, unit).unwrap();

        assert!(first < second);
    });
}

#[test]
fn with_millisecond_increases_after_2_milliseconds() {
    with_process(|process| {
        let unit = Atom::str_to_term("millisecond");

        let first = native(process, unit).unwrap();

        thread::sleep(Duration::from_millis(2));

        let second = native(process, unit).unwrap();

        assert!(first < second);
    });
}

#[test]
fn with_microsecond_increases_after_2_milliseconds() {
    with_process(|process| {
        let unit = Atom::str_to_term("microsecond");

        let first = native(process, unit).unwrap();

        thread::sleep(Duration::from_millis(2));

        let second = native(process, unit).unwrap();

        assert!(first < second);
    });
}

#[test]
fn with_nanosecond_increases_after_2_milliseconds() {
    with_process(|process| {
        let unit = Atom::str_to_term("nanosecond");

        let first = native(process, unit).unwrap();

        thread::sleep(Duration::from_millis(2));

        let second = native(process, unit).unwrap();

        assert!(first < second);
    });
}

#[test]
fn with_native_increases_after_2_native_time_units() {
    with_process(|process| {
        let unit = Atom::str_to_term("native");

        let first = native(process, unit).unwrap();

        thread::sleep(Duration::from_millis(2));

        let second = native(process, unit).unwrap();

        assert!(first < second);
    });
}

#[test]
fn with_perf_counter_increases_after_2_perf_counter_ticks() {
    with_process(|process| {
        let unit = Atom::str_to_term("perf_counter");

        let first = native(process, unit).unwrap();

        thread::sleep(Duration::from_millis(2));

        let second = native(process, unit).unwrap();

        assert!(first < second);
    });
}
