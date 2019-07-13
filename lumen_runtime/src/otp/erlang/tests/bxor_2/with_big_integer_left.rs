use super::*;

use std::mem::size_of;

use num_traits::Num;

#[test]
fn with_small_integer_right_returns_big_integer() {
    with(|left, process| {
        let right: Term = process.integer(0b1010_1010_1010_1010_1010_1010_1010);

        assert!(right.is_smallint());

        let result = erlang::bxor_2(left, right, &process);

        assert!(result.is_ok());

        let output = result.unwrap();

        assert!(output.is_bigint());
    });
}

#[test]
fn with_big_integer_right_returns_big_integer() {
    with(|left, process| {
        let right = process.integer(
            <BigInt as Num>::from_str_radix(
                "1010".repeat(size_of::<usize>() * (8 / 4) * 2).as_ref(),
                2,
            )
            .unwrap(),
        );

        assert!(right.is_bigint());

        let result = erlang::bxor_2(left, right, &process);

        assert!(result.is_ok());

        let output = result.unwrap();

        assert!(output.is_bigint());

        assert_eq!(
            output,
            process.integer(
                <BigInt as Num>::from_str_radix(
                    "0110".repeat(size_of::<usize>() * (8 / 4) * 2).as_ref(),
                    2
                )
                .unwrap()
            )
        );
    })
}

fn with<F>(f: F)
where
    F: FnOnce(Term, &ProcessControlBlock) -> (),
{
    with_process(|process| {
        let left = process.integer(
            <BigInt as Num>::from_str_radix(
                "1100".repeat(size_of::<usize>() * (8 / 4) * 2).as_ref(),
                2,
            )
            .unwrap(),
        );

        f(left, &process)
    })
}
