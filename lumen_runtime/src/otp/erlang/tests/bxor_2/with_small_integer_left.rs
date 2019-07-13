use super::*;

use std::mem::size_of;

use num_traits::Num;

#[test]
fn with_small_integer_right_returns_small_integer() {
    with_process(|process| {
        // all combinations of `0` and `1` bit.
        let left = process.integer(0b1100);
        let right = process.integer(0b1010);

        assert_eq!(
            erlang::bxor_2(left, right, &process),
            Ok(process.integer(0b0110))
        );
    })
}

#[test]
fn with_big_integer_right_returns_big_integer() {
    with_process(|process| {
        let left: Term = process.integer(0b1100_1100_1100_1100_1100_1100_1100_isize);

        assert!(left.is_smallint());

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
    })
}
