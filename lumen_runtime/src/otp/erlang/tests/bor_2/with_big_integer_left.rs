use super::*;

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
                    let result = erlang::bor_2(left, right, &arc_process);

                    prop_assert!(result.is_ok());

                    let bor = result.unwrap();

                    prop_assert!(bor.is_integer());

                    unsafe {
                        prop_assert!(left.count_ones() <= bor.count_ones());
                        prop_assert!(right.count_ones() <= bor.count_ones());
                    }

                    Ok(())
                },
            )
            .unwrap();
    });
}
