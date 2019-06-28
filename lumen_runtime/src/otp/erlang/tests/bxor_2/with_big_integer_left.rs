use super::*;

use std::mem::size_of;

use num_traits::Num;

#[test]
fn with_small_integer_right_returns_big_integer() {
    with(|left, process| {
        let right: Term = 0b1010_1010_1010_1010_1010_1010_1010.into_process(&process);

        assert_eq!(right.tag(), SmallInteger);

        let result = erlang::bxor_2(left, right, &process);

        assert!(result.is_ok());

        let output = result.unwrap();

        assert_eq!(output.tag(), Boxed);

        let unboxed_output: &Term = output.unbox_reference();

        assert_eq!(unboxed_output.tag(), BigInteger);
    });
}

#[test]
fn with_big_integer_right_returns_big_integer() {
    with(|left, process| {
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
        assert_eq!(
            output,
            <BigInt as Num>::from_str_radix(
                "0110".repeat(size_of::<usize>() * (8 / 4) * 2).as_ref(),
                2
            )
            .unwrap()
            .into_process(&process)
        );
    })
}

fn with<F>(f: F)
where
    F: FnOnce(Term, &Process) -> (),
{
    with_process(|process| {
        let left = <BigInt as Num>::from_str_radix(
            "1100".repeat(size_of::<usize>() * (8 / 4) * 2).as_ref(),
            2,
        )
        .unwrap()
        .into_process(&process);

        f(left, &process)
    })
}
