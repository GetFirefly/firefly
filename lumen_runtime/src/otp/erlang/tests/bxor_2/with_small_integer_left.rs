use super::*;

use std::mem::size_of;

use num_traits::Num;

#[test]
fn with_small_integer_right_returns_small_integer() {
    with_process(|process| {
        // all combinations of `0` and `1` bit.
        let left = 0b1100.into_process(&process);
        let right = 0b1010.into_process(&process);

        assert_eq!(
            erlang::bxor_2(left, right, &process),
            Ok(0b0110.into_process(&process))
        );
    })
}

#[test]
fn with_big_integer_right_returns_big_integer() {
    with_process(|process| {
        let left: Term = 0b1100_1100_1100_1100_1100_1100_1100_isize.into_process(&process);

        assert_eq!(left.tag(), SmallInteger);

        let right = <BigInt as Num>::from_str_radix(
            "1010".repeat(size_of::<usize>() * (8 / 4) * 2).as_ref(),
            2,
        )
        .unwrap()
        .into_process(&process);

        assert_eq!(right.tag(), Boxed);

        let unboxed_right: &Term = right.unbox_reference();

        assert_eq!(unboxed_right.tag(), BigInteger);

        let result = erlang::bxor_2(left, right, &process);

        assert!(result.is_ok());

        let output = result.unwrap();

        assert_eq!(output.tag(), Boxed);

        let unboxed_output: &Term = output.unbox_reference();

        assert_eq!(unboxed_output.tag(), BigInteger);
    })
}
