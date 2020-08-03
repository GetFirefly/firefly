use super::*;

use num_traits::Num;

use liblumen_alloc::erts::term::prelude::Term;

#[test]
fn with_negative_shifts_left_and_returns_big_integer() {
    with(|integer, process| {
        let shift = process.integer(-1).unwrap();

        assert_eq!(
            result(&process, integer, shift),
            Ok(process
                .integer(
                    <BigInt as Num>::from_str_radix(
                        "1011001110001111000011111000001111110000001111111000000011111111000000000",
                        2
                    )
                    .unwrap()
                )
                .unwrap())
        );
    });
}

#[test]
fn with_positive_with_big_integer_underflow_without_small_integer_underflow_returns_small_integer()
{
    with(|integer, process| {
        let shift = process.integer(71).unwrap();

        let result = result(&process, integer, shift);

        assert!(result.is_ok());

        let shifted = result.unwrap();

        assert!(shifted.is_smallint());
        assert_eq!(shifted, process.integer(0b1).unwrap());
    })
}

#[test]
fn with_positive_with_underflow_returns_zero() {
    with(|integer, process| {
        let shift = process.integer(80).unwrap();

        assert_eq!(
            result(&process, integer, shift),
            Ok(process.integer(0).unwrap())
        );
    });
}

fn with<F>(f: F)
where
    F: FnOnce(Term, &Process) -> (),
{
    with_process(|process| {
        let integer = process
            .integer(
                <BigInt as Num>::from_str_radix(
                    "101100111000111100001111100000111111000000111111100000001111111100000000",
                    2,
                )
                .unwrap(),
            )
            .unwrap();

        assert!(integer.is_boxed_bigint());

        f(integer, &process)
    })
}
