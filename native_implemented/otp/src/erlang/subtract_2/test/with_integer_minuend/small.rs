use super::*;

#[test]
fn with_small_integer_subtrahend_with_underflow_returns_big_integer() {
    with_process(|process| {
        let minuend = process.integer(Integer::MIN_SMALL).unwrap();
        let subtrahend = process.integer(Integer::MAX_SMALL).unwrap();

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
        let subtrahend = process.integer(Integer::MIN_SMALL).unwrap();

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
        let subtrahend = f64::MAX.into();

        assert_eq!(
            result(&process, minuend, subtrahend),
            Ok(f64::MIN.into())
        );
    })
}

#[test]
fn with_float_subtrahend_with_overflow_returns_max_float() {
    with(|minuend, process| {
        let subtrahend = f64::MIN.into();

        assert_eq!(
            result(&process, minuend, subtrahend),
            Ok(f64::MAX.into())
        );
    })
}

fn with<F>(f: F)
where
    F: FnOnce(Term, &Process) -> (),
{
    with_process(|process| {
        let minuend = process.integer(2).unwrap();

        f(minuend, &process)
    })
}
