use super::*;

#[test]
fn with_small_integer_subtrahend_with_underflow_returns_big_integer() {
    with_process(|process| {
        let minuend = crate::integer::small::MIN.into_process(&process);
        let subtrahend = crate::integer::small::MAX.into_process(&process);

        assert_eq!(subtrahend.tag(), SmallInteger);

        let result = erlang::subtract_2(minuend, subtrahend, &process);

        assert!(result.is_ok());

        let difference = result.unwrap();

        assert_eq!(difference.tag(), Boxed);

        let unboxed_difference: &Term = difference.unbox_reference();

        assert_eq!(unboxed_difference.tag(), BigInteger);
    })
}

#[test]
fn with_small_integer_subtrahend_with_overflow_returns_big_integer() {
    with(|minuend, process| {
        let subtrahend = crate::integer::small::MIN.into_process(&process);

        assert_eq!(subtrahend.tag(), SmallInteger);

        let result = erlang::subtract_2(minuend, subtrahend, &process);

        assert!(result.is_ok());

        let difference = result.unwrap();

        assert_eq!(difference.tag(), Boxed);

        let unboxed_difference: &Term = difference.unbox_reference();

        assert_eq!(unboxed_difference.tag(), BigInteger);
    })
}

#[test]
fn with_float_subtrahend_with_underflow_returns_min_float() {
    with(|minuend, process| {
        let subtrahend = std::f64::MAX.into_process(&process);

        assert_eq!(
            erlang::subtract_2(minuend, subtrahend, &process),
            Ok(std::f64::MIN.into_process(&process))
        );
    })
}

#[test]
fn with_float_subtrahend_with_overflow_returns_max_float() {
    with(|minuend, process| {
        let subtrahend = std::f64::MIN.into_process(&process);

        assert_eq!(
            erlang::subtract_2(minuend, subtrahend, &process),
            Ok(std::f64::MAX.into_process(&process))
        );
    })
}

fn with<F>(f: F)
where
    F: FnOnce(Term, &Process) -> (),
{
    with_process(|process| {
        let minuend = 2.into_process(&process);

        f(minuend, &process)
    })
}
