use super::*;

#[test]
fn without_integer_right_errors_badarith() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::integer::big(arc_process.clone()),
                    strategy::term::is_not_integer(arc_process.clone()),
                ),
                |(left, right)| {
                    prop_assert_eq!(erlang::band_2(left, right, &arc_process), Err(badarith!()));

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_integer_right_returns_bitwise_and() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::integer::big(arc_process.clone()),
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
