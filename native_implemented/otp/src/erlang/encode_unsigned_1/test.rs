use crate::erlang::encode_unsigned_1::result;
use crate::test::with_process;
use liblumen_alloc::erts::term::prelude::*;
use num_bigint::BigInt;
use crate::test::*;
use proptest::strategy::Just;

// 1> binary:encode_unsigned(11111111).
// <<169,138,199>>
#[test]
fn otp_doctest() {
    with_process(|process| {
        assert_eq!(
            result(process, process.integer(11111111)),
            Ok(process.binary_from_bytes(&[169,138,199]))
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
            Ok(process.binary_from_bytes(&[64]))
        )
    });
}

#[test]
fn not_integer() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_integer(arc_process.clone())
            )
        },
        |(arc_process, non_int)| {
            prop_assert_badarg!(
                result(&arc_process, non_int),
                "invalid integer conversion"
            );
            Ok(())
        },
    );
}