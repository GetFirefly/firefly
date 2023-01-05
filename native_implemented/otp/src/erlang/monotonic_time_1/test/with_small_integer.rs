use super::*;

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
        assert_badarg!(result(process, process.integer(0).unwrap()), SUPPORTED_UNITS);
    });
}

#[test]
fn with_positive_increases_after_2_time_units() {
    with_process(|process| {
        let unit = process.integer(2).unwrap();

        let first = result(process, unit).unwrap();

        let start_monotonic = monotonic::freeze();
        monotonic::freeze_at(start_monotonic + Duration::from_secs(2));

        let second = result(process, unit).unwrap();

        assert!(first < second);
    });
}
