use super::*;

#[test]
fn with_float_divisor_without_underflow_or_overflow_returns_float() {
    with(|dividend, process| {
        let divisor = process.float(4.0).unwrap();

        assert_eq!(
            erlang::divide_2(dividend, divisor, &process),
            Ok(process.float(0.5).unwrap())
        );
    })
}

#[test]
fn with_float_divisor_with_underflow_returns_min_float() {
    with_process(|process| {
        let dividend = process.float(std::f64::MIN).unwrap();
        let divisor = process.float(0.1).unwrap();

        assert_eq!(
            erlang::divide_2(dividend, divisor, &process),
            Ok(process.float(std::f64::MIN).unwrap())
        );
    })
}

#[test]
fn with_float_divisor_with_overflow_returns_max_float() {
    with_process(|process| {
        let dividend = process.float(std::f64::MAX).unwrap();
        let divisor = process.float(0.1).unwrap();

        assert_eq!(
            erlang::divide_2(dividend, divisor, &process),
            Ok(process.float(std::f64::MAX).unwrap())
        );
    })
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
