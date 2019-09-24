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
            Ok(process.integer(0b1000).unwrap())
        );
    })
}

#[test]
fn with_integer_right_returns_bitwise_and() {
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

                    let band = result.unwrap();

                    prop_assert!(band.is_integer());
                    prop_assert!(count_ones(band) <= count_ones(left));
                    prop_assert!(count_ones(band) <= count_ones(right));

                    Ok(())
                },
            )
            .unwrap();
    });
}
