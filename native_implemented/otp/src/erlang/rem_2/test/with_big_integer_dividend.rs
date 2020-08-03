use super::*;

use crate::test::with_big_int;

#[test]
fn with_small_integer_divisor_with_underflow_returns_small_integer() {
    with_big_int(|process, dividend| {
        let divisor: Term = process.integer(2).unwrap();

        assert!(divisor.is_smallint());

        let result = result(process, dividend, divisor);

        assert!(result.is_ok());

        let quotient = result.unwrap();

        assert!(quotient.is_smallint());
    })
}

#[test]
fn with_big_integer_divisor_with_underflow_returns_small_integer() {
    with_process(|process| {
        let dividend: Term = process.integer(SmallInteger::MAX_VALUE + 2).unwrap();
        let divisor: Term = process.integer(SmallInteger::MAX_VALUE + 1).unwrap();

        assert_eq!(
            result(process, dividend, divisor),
            Ok(process.integer(1).unwrap())
        );
    })
}

#[test]
fn with_big_integer_divisor_without_underflow_returns_big_integer() {
    with_process(|process| {
        let dividend: Term = process.integer(SmallInteger::MAX_VALUE + 1).unwrap();
        let divisor: Term = process.integer(SmallInteger::MAX_VALUE + 2).unwrap();

        assert_eq!(result(process, dividend, divisor), Ok(dividend));
    })
}
