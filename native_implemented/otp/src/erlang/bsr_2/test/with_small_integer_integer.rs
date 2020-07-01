use super::*;

use num_traits::Num;

use liblumen_alloc::erts::term::prelude::Term;

#[test]
fn with_negative_with_overflow_shifts_left_and_returns_big_integer() {
    with(|integer, process| {
        let shift = process.integer(-64).unwrap();

        assert_eq!(
            result(&process, integer, shift),
            Ok(process
                .integer(
                    <BigInt as Num>::from_str_radix(
                        "100000000000000000000000000000000000000000000000000000000000000000",
                        2
                    )
                    .unwrap()
                )
                .unwrap())
        );
    });
}

#[test]
fn with_negative_without_overflow_shifts_left_and_returns_small_integer() {
    with(|integer, process| {
        let shift = process.integer(-1).unwrap();

        assert_eq!(
            result(&process, integer, shift),
            Ok(process.integer(0b100).unwrap())
        );
    });
}

#[test]
fn with_positive_without_underflow_returns_small_integer() {
    with(|integer, process| {
        let shift = process.integer(1).unwrap();

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
        let shift = process.integer(3).unwrap();

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
        let integer = process.integer(0b10).unwrap();

        f(integer, &process)
    })
}
