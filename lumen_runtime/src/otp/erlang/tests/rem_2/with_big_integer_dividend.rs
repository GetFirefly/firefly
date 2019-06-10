use super::*;

#[test]
fn with_small_integer_divisor_with_underflow_returns_small_integer() {
    with(|dividend, process| {
        let divisor: Term = 2.into_process(&process);

        assert_eq!(divisor.tag(), SmallInteger);

        let result = erlang::rem_2(dividend, divisor, &process);

        assert!(result.is_ok());

        let quotient = result.unwrap();

        assert_eq!(quotient.tag(), SmallInteger);
    })
}

#[test]
fn with_big_integer_divisor_with_underflow_returns_small_integer() {
    with_process(|process| {
        let dividend: Term = (crate::integer::small::MAX + 2).into_process(&process);
        let divisor: Term = (crate::integer::small::MAX + 1).into_process(&process);

        assert_eq!(
            erlang::rem_2(dividend, divisor, &process),
            Ok(1.into_process(&process))
        );
    })
}

#[test]
fn with_big_integer_divisor_without_underflow_returns_big_integer() {
    with_process(|process| {
        let dividend: Term = (crate::integer::small::MAX + 1).into_process(&process);
        let divisor: Term = (crate::integer::small::MAX + 2).into_process(&process);

        assert_eq!(erlang::rem_2(dividend, divisor, &process), Ok(dividend));
    })
}

fn with<F>(f: F)
where
    F: FnOnce(Term, &Process) -> (),
{
    with_process(|process| {
        let dividend: Term = (crate::integer::small::MAX + 1).into_process(&process);

        assert_eq!(dividend.tag(), Boxed);

        let unboxed_dividend: &Term = dividend.unbox_reference();

        assert_eq!(unboxed_dividend.tag(), BigInteger);

        f(dividend, &process)
    })
}
