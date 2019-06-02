use super::*;

use num_traits::Num;

#[test]
fn with_negative_without_underflow_shifts_right() {
    with_process(|process| {
        let integer = 0b101100111000.into_process(&process);
        let shift = (-9).into_process(&process);

        assert_eq!(
            erlang::bsl_2(integer, shift, &process),
            Ok(0b101.into_process(&process))
        );
    });
}

#[test]
fn with_negative_with_underflow_returns_zero() {
    with_process(|process| {
        let integer = 0b101100111000.into_process(&process);
        let shift = (-12).into_process(&process);

        assert_eq!(
            erlang::bsl_2(integer, shift, &process),
            Ok(0b0.into_process(&process))
        );
    });
}

#[test]
fn with_positive_without_overflow_returns_small_integer() {
    with_process(|process| {
        let integer = 0b1.into_process(&process);
        let shift = 1.into_process(&process);

        let result = erlang::bsl_2(integer, shift, &process);

        assert!(result.is_ok());

        let shifted = result.unwrap();

        assert_eq!(shifted.tag(), SmallInteger);
        assert_eq!(shifted, 0b10.into_process(&process));
    })
}

#[test]
fn with_positive_with_overflow_returns_big_integer() {
    with_process(|process| {
        let integer = 0b1.into_process(&process);
        let shift = 64.into_process(&process);

        let result = erlang::bsl_2(integer, shift, &process);

        assert!(result.is_ok());

        let shifted = result.unwrap();

        assert_eq!(shifted.tag(), Boxed);

        let unboxed_shifted: &Term = shifted.unbox_reference();

        assert_eq!(unboxed_shifted.tag(), BigInteger);
        assert_eq!(
            shifted,
            <BigInt as Num>::from_str_radix("18446744073709551616", 10)
                .unwrap()
                .into_process(&process)
        );
    });
}
