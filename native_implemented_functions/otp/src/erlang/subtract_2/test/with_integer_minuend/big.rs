use super::*;

use crate::test::with_big_int;

#[test]
fn with_big_integer_subtrahend_with_underflow_returns_small_integer() {
    with_big_int(|process, minuend| {
        let subtrahend = process.integer(SmallInteger::MAX_VALUE + 1).unwrap();

        assert!(subtrahend.is_boxed_bigint());

        let result = result(&process, minuend, subtrahend);

        assert!(result.is_ok());

        let difference = result.unwrap();

        assert!(difference.is_smallint());
    })
}

#[test]
fn with_float_subtrahend_with_underflow_returns_min_float() {
    with_big_int(|process, minuend| {
        let subtrahend = process.float(std::f64::MAX).unwrap();

        assert_eq!(
            result(&process, minuend, subtrahend),
            Ok(process.float(std::f64::MIN).unwrap())
        );
    })
}

#[test]
fn with_float_subtrahend_with_overflow_returns_max_float() {
    with_big_int(|process, minuend| {
        let subtrahend = process.float(std::f64::MIN).unwrap();

        assert_eq!(
            result(&process, minuend, subtrahend),
            Ok(process.float(std::f64::MAX).unwrap())
        );
    })
}
