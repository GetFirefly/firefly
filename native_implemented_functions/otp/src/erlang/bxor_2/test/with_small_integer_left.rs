use super::*;

use std::mem::size_of;

use num_traits::Num;

#[test]
fn with_small_integer_right_returns_small_integer() {
    with_process(|process| {
        // all combinations of `0` and `1` bit.
        let left = process.integer(0b1100).unwrap();
        let right = process.integer(0b1010).unwrap();

        assert_eq!(
            result(&process, left, right),
            Ok(process.integer(0b0110).unwrap())
        );
    })
}

#[test]
fn with_big_integer_right_returns_big_integer() {
    with_process(|process| {
        let left = process
            .integer(0b1100_1100_1100_1100_1100_1100_1100_isize)
            .unwrap();

        assert!(left.is_smallint());

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
    })
}
