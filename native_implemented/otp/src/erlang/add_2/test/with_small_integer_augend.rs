use super::*;

#[test]
fn without_number_addend_errors_badarith() {
    super::without_number_addend_errors_badarith(file!(), strategy::term::integer::small);
}

// `with_small_integer_addend_without_underflow_or_overflow_returns_small_integer` in integration
// tests

#[test]
fn with_small_integer_addend_with_underflow_returns_big_integer() {
    with_process(|process| {
        let augend = process.integer(-1_isize);
        let addend = process.integer(SmallInteger::MIN_VALUE);

        assert!(addend.is_smallint());

        let result = result(&process, augend, addend);

        assert!(result.is_ok());

        let sum = result.unwrap();

        assert!(sum.is_boxed_bigint());
    })
}

// `with_small_integer_addend_with_overflow_returns_big_integer` in integration tests
// `with_big_integer_addend_returns_big_integer` in integration tests
// `with_float_addend_without_underflow_or_overflow_returns_float` in integration tests

#[test]
fn with_float_addend_with_underflow_returns_min_float() {
    with_process(|process| {
        let augend = process.integer(-1);
        let addend = process.float(std::f64::MIN);

        assert_eq!(
            result(&process, augend, addend),
            Ok(process.float(std::f64::MIN))
        );
    })
}
