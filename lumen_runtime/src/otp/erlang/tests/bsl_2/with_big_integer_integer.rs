use super::*;

use num_traits::Num;

#[test]
fn with_negative_without_big_integer_underflow_shifts_right_and_returns_big_integer() {
    with(|integer, process| {
        let shift = (-1).into_process(&process);

        let result = erlang::bsl_2(integer, shift, &process);

        assert!(result.is_ok());

        let shifted = result.unwrap();

        assert_eq!(shifted.tag(), Boxed);

        let unboxed_shifted: &Term = shifted.unbox_reference();

        assert_eq!(unboxed_shifted.tag(), BigInteger);

        assert_eq!(
            shifted,
            <BigInt as Num>::from_str_radix(
                "10110011100011110000111110000011111100000011111110000000111111110000000",
                2
            )
            .unwrap()
            .into_process(&process)
        );
    });
}

#[test]
fn with_negative_with_big_integer_underflow_without_small_integer_underflow_shifts_right_and_returns_small_integer(
) {
    with(|integer, process| {
        let shift = (-69).into_process(&process);

        let result = erlang::bsl_2(integer, shift, &process);

        assert!(result.is_ok());

        let shifted = result.unwrap();

        assert_eq!(shifted.tag(), SmallInteger);
        assert_eq!(shifted, 0b101.into_process(&process));
    });
}

#[test]
fn with_negative_with_underflow_returns_zero() {
    with(|integer, process| {
        let shift = (-74).into_process(&process);

        assert_eq!(
            erlang::bsl_2(integer, shift, &process),
            Ok(0b0.into_process(&process))
        );
    });
}

#[test]
fn with_positive_returns_big_integer() {
    with(|integer, process| {
        let shift = 1.into_process(&process);

        let result = erlang::bsl_2(integer, shift, &process);

        assert!(result.is_ok());

        let shifted = result.unwrap();

        assert_eq!(shifted.tag(), Boxed);

        let unboxed_shifted: &Term = shifted.unbox_reference();

        assert_eq!(unboxed_shifted.tag(), BigInteger);
        assert_eq!(
            shifted,
            <BigInt as Num>::from_str_radix(
                "1011001110001111000011111000001111110000001111111000000011111111000000000",
                2
            )
            .unwrap()
            .into_process(&process)
        );
    });
}

fn with<F>(f: F)
where
    F: FnOnce(Term, &Process) -> (),
{
    with_process(|process| {
        let integer = <BigInt as Num>::from_str_radix(
            "101100111000111100001111100000111111000000111111100000001111111100000000",
            2,
        )
        .unwrap()
        .into_process(&process);

        assert_eq!(integer.tag(), Boxed);

        let integer_unboxed: &Term = integer.unbox_reference();

        assert_eq!(integer_unboxed.tag(), BigInteger);

        f(integer, &process)
    })
}
