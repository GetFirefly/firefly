use super::*;

use std::thread;
use std::time::Duration;

#[test]
fn with_negative_errors_badarg() {
    with_process(|process| {
        assert_badarg!(
            result(process, process.integer(-1).unwrap()),
            "hertz must be positive"
        );
    });
}

#[test]
fn with_zero_errors_badarg() {
    with_process(|process| {
        assert_badarg!(
            result(process, process.integer(0).unwrap()),
            "hertz must be positive"
        );
    });
}

#[test]
fn with_positive_increases_after_2_time_units() {
    with_process(|process| {
        let unit = process.integer(2).unwrap();

        let first = result(process, unit).unwrap();

        thread::sleep(Duration::from_secs(1));

        let second = result(process, unit).unwrap();

        assert!(first < second);
    });
}
