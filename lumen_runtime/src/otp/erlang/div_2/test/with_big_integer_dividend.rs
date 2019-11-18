use super::*;

#[test]
fn with_small_integer_divisor_with_underflow_returns_small_integer() {
    with(|dividend, process| {
        let divisor = process.integer(2).unwrap();

        assert!(divisor.is_smallint());

        let result = native(process, dividend, divisor);

        assert!(result.is_ok());

        let quotient = result.unwrap();

        assert!(quotient.is_smallint());
    })
}

#[test]
fn with_big_integer_divisor_with_underflow_returns_small_integer() {
    with(|dividend, process| {
        let divisor = dividend;

        assert!(divisor.is_boxed_bigint());

        let result = native(process, dividend, divisor);

        assert!(result.is_ok());

        let quotient = result.unwrap();

        assert!(quotient.is_smallint());
        assert_eq!(quotient, process.integer(1).unwrap())
    })
}

#[test]
fn with_big_integer_divisor_without_underflow_returns_big_integer() {
    with_process(|process| {
        let divisor = process.integer(SmallInteger::MAX_VALUE + 1).unwrap();

        assert!(divisor.is_boxed_bigint());

        let dividend = erlang::multiply_2::native(&process, divisor, divisor).unwrap();

        let result = native(process, dividend, divisor);

        assert!(result.is_ok());

        let quotient = result.unwrap();

        assert!(quotient.is_boxed_bigint());
        assert_eq!(quotient, divisor);
    })
}

fn with<F>(f: F)
where
    F: FnOnce(Term, &Process) -> (),
{
    with_process(|process| {
        let dividend: Term = process.integer(SmallInteger::MAX_VALUE + 1).unwrap();

        assert!(dividend.is_boxed_bigint());

        f(dividend, &process)
    })
}
