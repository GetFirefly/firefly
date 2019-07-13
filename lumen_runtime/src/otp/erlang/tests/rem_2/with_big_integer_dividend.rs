use super::*;

#[test]
fn with_small_integer_divisor_with_underflow_returns_small_integer() {
    with(|dividend, process| {
        let divisor: Term = process.integer(2);

        assert!(divisor.is_smallint());

        let result = erlang::rem_2(dividend, divisor, &process);

        assert!(result.is_ok());

        let quotient = result.unwrap();

        assert!(quotient.is_smallint());
    })
}

#[test]
fn with_big_integer_divisor_with_underflow_returns_small_integer() {
    with_process(|process| {
        let dividend: Term = process.integer(SmallInteger::MAX_VALUE + 2);
        let divisor: Term = process.integer(SmallInteger::MAX_VALUE + 1);

        assert_eq!(
            erlang::rem_2(dividend, divisor, &process),
            Ok(process.integer(1))
        );
    })
}

#[test]
fn with_big_integer_divisor_without_underflow_returns_big_integer() {
    with_process(|process| {
        let dividend: Term = process.integer(SmallInteger::MAX_VALUE + 1);
        let divisor: Term = process.integer(SmallInteger::MAX_VALUE + 2);

        assert_eq!(erlang::rem_2(dividend, divisor, &process), Ok(dividend));
    })
}

fn with<F>(f: F)
where
    F: FnOnce(Term, &ProcessControlBlock) -> (),
{
    with_process(|process| {
        let dividend: Term = process.integer(SmallInteger::MAX_VALUE + 1);

        assert!(dividend.is_bigint());

        f(dividend, &process)
    })
}
