use super::*;

#[test]
fn with_float_divisor_without_underflow_or_overflow_returns_float() {
    with(|dividend, process| {
        let divisor = 4.0.into_process(&process);

        assert_eq!(
            erlang::divide_2(dividend, divisor, &process),
            Ok(0.5.into_process(&process))
        );
    })
}

#[test]
fn with_float_divisor_with_underflow_returns_min_float() {
    with_process(|process| {
        let dividend = std::f64::MIN.into_process(&process);
        let divisor = 0.1.into_process(&process);

        assert_eq!(
            erlang::divide_2(dividend, divisor, &process),
            Ok(std::f64::MIN.into_process(&process))
        );
    })
}

#[test]
fn with_float_divisor_with_overflow_returns_max_float() {
    with_process(|process| {
        let dividend = std::f64::MAX.into_process(&process);
        let divisor = 0.1.into_process(&process);

        assert_eq!(
            erlang::divide_2(dividend, divisor, &process),
            Ok(std::f64::MAX.into_process(&process))
        );
    })
}

fn with<F>(f: F)
where
    F: FnOnce(Term, &Process) -> (),
{
    with_process(|process| {
        let dividend = 2.0.into_process(&process);

        f(dividend, &process)
    })
}
