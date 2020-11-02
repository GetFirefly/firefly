use crate::binary::encode_unsigned_1::result;
use crate::test::with_process;
use crate::test::*;
use liblumen_alloc::erts::term::prelude::*;
use num_bigint::{BigInt, ToBigInt};
use proptest::strategy::Just;

// 1> binary:encode_unsigned(11111111).
// <<169,138,199>>
#[test]
fn otp_doctest() {
    with_process(|process| {
        assert_eq!(
            result(process, process.integer(11111111)),
            Ok(process.binary_from_bytes(&[169, 138, 199]))
        )
    });
}

#[test]
fn smallest_big_int() {
    let largest_small_int_as_big_int: BigInt = SmallInteger::MAX_VALUE.into();
    let smallest_big_int: BigInt = largest_small_int_as_big_int + 1;

    // 1> binary:encode_unsigned(70368744177664).
    // <<64,0,0,0,0,0>>

    with_process(|process| {
        assert_eq!(
            result(process, process.integer(smallest_big_int)),
            Ok(process.binary_from_bytes(&[64, 0, 0, 0, 0, 0]))
        )
    });
}

#[test]
fn big_int_with_middle_zeros() {
    let largest_small_int_as_big_int: BigInt = SmallInteger::MAX_VALUE.into();
    let big_int_with_middle_zeros: BigInt = largest_small_int_as_big_int + 2;

    // 1> binary:encode_unsigned(70368744177665).
    // <<64,0,0,0,0,1>>
    with_process(|process| {
        assert_eq!(
            result(process, process.integer(big_int_with_middle_zeros)),
            Ok(process.binary_from_bytes(&[64, 0, 0, 0, 0, 1]))
        )
    });
}

#[test]
fn small_int_with_middle_zeros() {
    // 1> binary:encode_unsigned(11075783).
    // <<169,0,199>>
    let largest_small_int_as_big_int: BigInt = SmallInteger::MAX_VALUE.into();
    assert!(11075783.to_bigint().unwrap() < largest_small_int_as_big_int);

    with_process(|process| {
        assert_eq!(
            result(process, process.integer(11075783)),
            Ok(process.binary_from_bytes(&[169, 0, 199]))
        )
    });
}

#[test]
fn small_int_with_trailing_zeros() {
    // 1> binary:encode_unsigned(16777216).
    // <<1,0,0,0>>
    let largest_small_int_as_big_int: BigInt = SmallInteger::MAX_VALUE.into();
    assert!(16777216.to_bigint().unwrap() < largest_small_int_as_big_int);

    with_process(|process| {
        assert_eq!(
            result(process, process.integer(16777216)),
            Ok(process.binary_from_bytes(&[1, 0, 0, 0]))
        )
    });
}

#[test]
fn negative_integer() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::integer::negative(arc_process.clone()),
            )
        },
        |(arc_process, non_int)| {
            prop_assert_badarg!(result(&arc_process, non_int), "invalid integer conversion");
            Ok(())
        },
    );
}

#[test]
fn not_integer() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_integer(arc_process.clone()),
            )
        },
        |(arc_process, non_int)| {
            prop_assert_badarg!(result(&arc_process, non_int), "invalid integer conversion");
            Ok(())
        },
    );
}
