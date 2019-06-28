use super::*;

use num_traits::Num;

#[test]
fn with_negative_shifts_left_and_returns_big_integer() {
    with(|integer, process| {
        let shift = (-1).into_process(&process);

        assert_eq!(
            erlang::bsr_2(integer, shift, &process),
            Ok(<BigInt as Num>::from_str_radix(
                "1011001110001111000011111000001111110000001111111000000011111111000000000",
                2
            )
            .unwrap()
            .into_process(&process))
        );
    });
}

#[test]
fn with_positive_with_big_integer_underflow_without_small_integer_underflow_returns_small_integer()
{
    with(|integer, process| {
        let shift = 71.into_process(&process);

        let result = erlang::bsr_2(integer, shift, &process);

        assert!(result.is_ok());

        let shifted = result.unwrap();

        assert_eq!(shifted.tag(), SmallInteger);
        assert_eq!(shifted, 0b1.into_process(&process));
    })
}

#[test]
fn with_positive_with_underflow_returns_zero() {
    with(|integer, process| {
        let shift = 80.into_process(&process);

        assert_eq!(
            erlang::bsr_2(integer, shift, &process),
            Ok(0.into_process(&process))
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
