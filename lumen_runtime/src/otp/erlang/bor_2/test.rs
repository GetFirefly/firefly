mod with_big_integer_left;
mod with_small_integer_left;

use proptest::test_runner::{Config, TestCaseError, TestRunner};
use proptest::{prop_assert, prop_assert_eq};

use liblumen_alloc::erts::term::prelude::{Encoded, TypedTerm};

use crate::otp::erlang::bor_2::native;
use crate::scheduler::{with_process, with_process_arc};
use crate::test::{count_ones, strategy};

#[test]
fn without_integer_left_errors_badarith() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_not_integer(arc_process.clone()),
                    strategy::term::is_integer(arc_process.clone()),
                ),
                |(left, right)| {
                    prop_assert_badarith!(
                        native(&arc_process, left, right),
                        format!(
                            "left_integer ({}) and right_integer ({}) are not both integers",
                            left, right
                        )
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_integer_left_without_integer_right_errors_badarith() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_integer(arc_process.clone()),
                    strategy::term::is_not_integer(arc_process.clone()),
                ),
                |(left, right)| {
                    prop_assert_badarith!(
                        native(&arc_process, left, right),
                        format!(
                            "left_integer ({}) and right_integer ({}) are not both integers",
                            left, right
                        )
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_same_integer_returns_same_integer() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_integer(arc_process.clone()),
                |operand| {
                    prop_assert_eq!(native(&arc_process, operand, operand), Ok(operand));

                    Ok(())
                },
            )
            .unwrap();
    });
}

fn typed_count_ones(term: &TypedTerm) -> u32 {
    match term {
        &TypedTerm::BigInteger(big) => big.as_ref().count_ones(),
        &TypedTerm::SmallInteger(small) => small.count_ones(),
        _ => unreachable!(),
    }
}

fn get_bytes(term: &TypedTerm) -> Vec<u8> {
    match term {
        &TypedTerm::BigInteger(big) => big.as_ref().to_signed_bytes_le(),
        &TypedTerm::SmallInteger(small) => small.to_le_bytes(),
        _ => unreachable!(),
    }
}

fn is_correct_bit_representation(
    counted: u32,
    mut lhs: Vec<u8>,
    mut rhs: Vec<u8>,
) -> Result<(), TestCaseError> {
    let lhs_len = lhs.len();
    let rhs_len = rhs.len();

    // Sign-extend if necessary
    if lhs_len < rhs_len {
        // Last byte will have its high bit set
        let is_neg = lhs[lhs_len - 1] >= 128;
        let extend_bytes = rhs_len - lhs_len;
        let mut temp = Vec::new();
        if is_neg {
            temp.resize(extend_bytes, 255u8);
        } else {
            temp.resize(extend_bytes, 0u8);
        }
        temp.extend_from_slice(lhs.as_slice());
        lhs = temp;
    } else if rhs_len < lhs_len {
        let is_neg = rhs[rhs_len - 1] >= 128;
        let extend_bytes = lhs_len - rhs_len;
        let mut temp = Vec::new();
        if is_neg {
            temp.resize(extend_bytes, 255u8);
        } else {
            temp.resize(extend_bytes, 0u8);
        }
        temp.extend_from_slice(rhs.as_slice());
        rhs = temp;
    }

    // Count the total number of ones when OR'd
    let count = lhs
        .iter()
        .copied()
        .zip(rhs.iter().copied())
        .map(|(l, r)| (l | r).count_ones())
        .sum();

    // If the counts differ, we probably have a compacted
    // representation, so make sure the modulus in bits is
    // the same for both, which implies that the 1-bit count
    // would be the same if the original had been sign-extended
    if counted != count {
        let diff = counted % 8;
        prop_assert_eq!((count - diff) % 8, 0);
        prop_assert_eq!((counted - diff) % 8, 0);
    } else {
        prop_assert_eq!(counted, count);
    }
    Ok(())
}
