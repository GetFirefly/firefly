use super::*;

// No using `proptest` because I'd rather cover truth table manually
#[test]
fn with_small_integer_right_returns_small_integer() {
    with_process(|process| {
        // all combinations of `0` and `1` bit.
        let left = process.integer(0b1100).unwrap();
        let right = process.integer(0b1010).unwrap();

        assert_eq!(
            native(&process, left, right),
            Ok(process.integer(0b1110).unwrap())
        );
    })
}

#[test]
fn with_integer_right_returns_bitwise_or() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::integer::small(arc_process.clone()),
                    strategy::term::is_integer(arc_process.clone()),
                ),
                |(left, right)| {
                    let result = native(&arc_process, left, right);

                    prop_assert!(result.is_ok());

                    let bor = result.unwrap();

                    prop_assert!(bor.is_integer());
                    prop_assert!(count_ones(left) <= count_ones(bor));
                    prop_assert!(count_ones(right) <= count_ones(bor));

                    Ok(())
                },
            )
            .unwrap();
    });
}
