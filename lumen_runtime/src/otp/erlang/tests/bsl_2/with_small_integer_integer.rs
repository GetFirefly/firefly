use super::*;

use num_traits::Num;

#[test]
fn with_negative_without_underflow_shifts_right() {
    with_process(|process| {
        let integer = process.integer(0b101100111000);
        let shift = process.integer(-9);

        assert_eq!(
            erlang::bsl_2(integer, shift, &process),
            Ok(process.integer(0b101))
        );
    });
}

#[test]
fn with_negative_with_underflow_returns_zero() {
    with_process(|process| {
        let integer = process.integer(0b101100111000);
        let shift = process.integer(-12);

        assert_eq!(
            erlang::bsl_2(integer, shift, &process),
            Ok(process.integer(0b0))
        );
    });
}

#[test]
fn with_positive_without_overflow_returns_small_integer() {
    with_process(|process| {
        let integer = process.integer(0b1);
        let shift = process.integer(1);

        let result = erlang::bsl_2(integer, shift, &process);

        assert!(result.is_ok());

        let shifted = result.unwrap();

        assert!(shifted.is_smallint());
        assert_eq!(shifted, process.integer(0b10));
    })
}

#[test]
fn with_positive_with_overflow_returns_big_integer() {
    with_process(|process| {
        let integer = process.integer(0b1);
        let shift = process.integer(64);

        let result = erlang::bsl_2(integer, shift, &process);

        assert!(result.is_ok());

        let shifted = result.unwrap();

        assert!(shifted.is_bigint());

        assert_eq!(
            shifted,
            process.integer(<BigInt as Num>::from_str_radix("18446744073709551616", 10).unwrap())
        );
    });
}
