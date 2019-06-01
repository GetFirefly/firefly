use super::*;

// No using `proptest` because I'd rather cover truth table manually
#[test]
fn with_small_integer_right_returns_small_integer() {
    with_process(|process| {
        // all combinations of `0` and `1` bit.
        let left = 0b1100.into_process(&process);
        let right = 0b1010.into_process(&process);

        assert_eq!(
            erlang::band_2(left, right, &process),
            Ok(0b1000.into_process(&process))
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
                    let result = erlang::band_2(left, right, &arc_process);

                    prop_assert!(result.is_ok());

                    let band = result.unwrap();

                    prop_assert!(band.is_integer());

                    unsafe {
                        prop_assert!(band.count_ones() <= left.count_ones());
                        prop_assert!(band.count_ones() <= right.count_ones());
                    }

                    Ok(())
                },
            )
            .unwrap();
    });
}
