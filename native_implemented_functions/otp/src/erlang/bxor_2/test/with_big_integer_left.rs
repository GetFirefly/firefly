use super::*;

use std::mem::size_of;

use num_traits::Num;

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::Term;

#[test]
fn with_small_integer_right_returns_big_integer() {
    with(|left, process| {
        let right = process
            .integer(0b1010_1010_1010_1010_1010_1010_1010)
            .unwrap();

        assert!(right.is_smallint());

        let result = result(&process, left, right);

        assert!(result.is_ok());

        let output = result.unwrap();

        assert!(output.is_boxed_bigint());
    });
}

#[test]
fn with_big_integer_right_returns_big_integer() {
    with(|left, process| {
        let right = process
            .integer(
                <BigInt as Num>::from_str_radix(
                    "1010".repeat(size_of::<usize>() * (8 / 4) * 2).as_ref(),
                    2,
                )
                .unwrap(),
            )
            .unwrap();

        assert!(right.is_boxed_bigint());

        let result = result(&process, left, right);

        assert!(result.is_ok());

        let output = result.unwrap();

        assert!(output.is_boxed_bigint());

        assert_eq!(
            output,
            process
                .integer(
                    <BigInt as Num>::from_str_radix(
                        "0110".repeat(size_of::<usize>() * (8 / 4) * 2).as_ref(),
                        2
                    )
                    .unwrap()
                )
                .unwrap()
        );
    })
}

fn with<F>(f: F)
where
    F: FnOnce(Term, &Process) -> (),
{
    with_process(|process| {
        let left = process
            .integer(
                <BigInt as Num>::from_str_radix(
                    "1100".repeat(size_of::<usize>() * (8 / 4) * 2).as_ref(),
                    2,
                )
                .unwrap(),
            )
            .unwrap();

        f(left, &process)
    })
}
