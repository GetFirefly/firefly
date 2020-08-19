use super::*;

#[test]
fn with_small_integer_subtrahend_with_underflow_returns_big_integer() {
    with_process(|process| {
        let minuend = process.integer(SmallInteger::MIN_VALUE);
        let subtrahend = process.integer(SmallInteger::MAX_VALUE);

        assert!(subtrahend.is_smallint());

        let result = result(&process, minuend, subtrahend);

        assert!(result.is_ok());

        let difference = result.unwrap();

        assert!(difference.is_boxed_bigint());
    })
}

#[test]
fn with_small_integer_subtrahend_with_overflow_returns_big_integer() {
    with(|minuend, process| {
        let subtrahend = process.integer(SmallInteger::MIN_VALUE);

        assert!(subtrahend.is_smallint());

        let result = result(&process, minuend, subtrahend);

        assert!(result.is_ok());

        let difference = result.unwrap();

        assert!(difference.is_boxed_bigint());
    })
}

#[test]
fn with_float_subtrahend_with_underflow_returns_min_float() {
    with(|minuend, process| {
        let subtrahend = process.float(std::f64::MAX);

        assert_eq!(
            result(&process, minuend, subtrahend),
            Ok(process.float(std::f64::MIN))
        );
    })
}

#[test]
fn with_float_subtrahend_with_overflow_returns_max_float() {
    with(|minuend, process| {
        let subtrahend = process.float(std::f64::MIN);

        assert_eq!(
            result(&process, minuend, subtrahend),
            Ok(process.float(std::f64::MAX))
        );
    })
}

fn with<F>(f: F)
where
    F: FnOnce(Term, &Process) -> (),
{
    with_process(|process| {
        let minuend = process.integer(2);

        f(minuend, &process)
    })
}
