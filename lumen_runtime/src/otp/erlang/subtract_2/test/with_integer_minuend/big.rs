use super::*;

#[test]
fn with_big_integer_subtrahend_with_underflow_returns_small_integer() {
    with(|minuend, process| {
        let subtrahend = process.integer(SmallInteger::MAX_VALUE + 1).unwrap();

        assert!(subtrahend.is_bigint());

        let result = native(&process, minuend, subtrahend);

        assert!(result.is_ok());

        let difference = result.unwrap();

        assert!(difference.is_smallint());
    })
}

#[test]
fn with_float_subtrahend_with_underflow_returns_min_float() {
    with(|minuend, process| {
        let subtrahend = process.float(std::f64::MAX).unwrap();

        assert_eq!(
            native(&process, minuend, subtrahend),
            Ok(process.float(std::f64::MIN).unwrap())
        );
    })
}

#[test]
fn with_float_subtrahend_with_overflow_returns_max_float() {
    with(|minuend, process| {
        let subtrahend = process.float(std::f64::MIN).unwrap();

        assert_eq!(
            native(&process, minuend, subtrahend),
            Ok(process.float(std::f64::MAX).unwrap())
        );
    })
}

fn with<F>(f: F)
where
    F: FnOnce(Term, &ProcessControlBlock) -> (),
{
    with_process(|process| {
        let minuend: Term = process.integer(SmallInteger::MAX_VALUE + 1).unwrap();

        assert!(minuend.is_bigint());

        f(minuend, &process)
    })
}
