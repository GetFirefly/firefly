use super::*;

use num_traits::Num;

#[test]
fn with_negative_with_overflow_shifts_left_and_returns_big_integer() {
    with(|integer, process| {
        let shift = (-64).into_process(&process);

        assert_eq!(
            erlang::bsr_2(integer, shift, &process),
            Ok(<BigInt as Num>::from_str_radix(
                "100000000000000000000000000000000000000000000000000000000000000000",
                2
            )
            .unwrap()
            .into_process(&process))
        );
    });
}

#[test]
fn with_negative_without_overflow_shifts_left_and_returns_small_integer() {
    with(|integer, process| {
        let shift = (-1).into_process(&process);

        assert_eq!(
            erlang::bsr_2(integer, shift, &process),
            Ok(0b100.into_process(&process))
        );
    });
}

#[test]
fn with_positive_without_underflow_returns_small_integer() {
    with(|integer, process| {
        let shift = 1.into_process(&process);

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
        let shift = 3.into_process(&process);

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
        let integer = 0b10.into_process(&process);

        f(integer, &process)
    })
}
