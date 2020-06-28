use super::*;

// No using `proptest` because I'd rather cover truth table manually
#[test]
fn with_small_integer_right_returns_small_integer() {
    with_process(|process| {
        // all combinations of `0` and `1` bit.
        let left = process.integer(0b1100).unwrap();
        let right = process.integer(0b1010).unwrap();

        assert_eq!(
            result(&process, left, right),
            Ok(process.integer(0b1110).unwrap())
        );
    })
}

#[test]
fn with_integer_right_returns_bitwise_or() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::integer::small(arc_process.clone()),
                strategy::term::is_integer(arc_process.clone()),
            )
        },
        |(arc_process, left, right)| {
            let result = result(&arc_process, left, right);

            prop_assert!(result.is_ok());

            let bor = result.unwrap();

            prop_assert!(bor.is_integer());
            // BigInt compacts its representation when possible,
            // which means that the number of 1 bits in the output
            // will not necessarily hold the property that bitwise
            // OR always produces strictly the same or greater number
            // of 1s in the output.
            match bor.decode().unwrap() {
                TypedTerm::BigInteger(big_bor) => {
                    let bor_count = big_bor.count_ones();
                    let lhs = left.decode().unwrap();
                    let rhs = right.decode().unwrap();

                    let lhs_count = typed_count_ones(&lhs);
                    let rhs_count = typed_count_ones(&rhs);
                    if bor_count < lhs_count || bor_count < rhs_count {
                        // We have a compacted output, so we need to verify the input
                        let lhs_bytes = self::get_bytes(&lhs);
                        let rhs_bytes = self::get_bytes(&rhs);
                        is_correct_bit_representation(bor_count, lhs_bytes, rhs_bytes)?;
                    } else {
                        // Our standard check works here
                        prop_assert!(lhs_count <= bor_count);
                        prop_assert!(rhs_count <= bor_count);
                    }
                }
                TypedTerm::SmallInteger(_) => {
                    // Use the standard check
                    prop_assert!(count_ones(left) <= count_ones(bor));
                    prop_assert!(count_ones(right) <= count_ones(bor));
                }
                _ => unreachable!(),
            }

            Ok(())
        },
    );
}
