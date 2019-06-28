use super::*;

#[test]
fn with_big_integer_subtrahend_with_underflow_returns_small_integer() {
    with(|minuend, process| {
        let subtrahend = (crate::integer::small::MAX + 1).into_process(&process);

        assert_eq!(subtrahend.tag(), Boxed);

        let unboxed_subtrahend: &Term = subtrahend.unbox_reference();

        assert_eq!(unboxed_subtrahend.tag(), BigInteger);

        let result = erlang::subtract_2(minuend, subtrahend, &process);

        assert!(result.is_ok());

        let difference = result.unwrap();

        assert_eq!(difference.tag(), SmallInteger);
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
        let minuend: Term = (crate::integer::small::MAX + 1).into_process(&process);

        assert_eq!(minuend.tag(), Boxed);

        let unboxed_minuend: &Term = minuend.unbox_reference();

        assert_eq!(unboxed_minuend.tag(), BigInteger);

        f(minuend, &process)
    })
}
