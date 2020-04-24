use super::*;

#[test]
fn with_float_divisor_without_underflow_or_overflow_returns_float() {
    with(|dividend, process| {
        let divisor = process.float(4.0).unwrap();

        assert_eq!(
            result(process, dividend, divisor),
            Ok(process.float(0.5).unwrap())
        );
    })
}

#[test]
fn with_float_divisor_with_underflow_returns_min_float() {
    with_extreme(std::f64::MIN);
}

#[test]
fn with_float_divisor_with_overflow_returns_max_float() {
    with_extreme(std::f64::MAX);
}

fn with<F>(f: F)
where
    F: FnOnce(Term, &Process) -> (),
{
    with_process(|process| {
        let dividend = process.float(2.0).unwrap();

        f(dividend, &process)
    })
}

fn with_extreme(extreme: f64) {
    with_process(|process| {
        let dividend = process.float(extreme).unwrap();
        let divisor = process.float(0.1).unwrap();

        assert_eq!(
            result(process, dividend, divisor),
            Ok(process.float(extreme).unwrap())
        );
    })
}
