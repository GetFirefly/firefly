use super::*;

mod with_big_integer_left;
mod with_small_integer_left;

#[test]
fn without_integer_right_errors_badarith() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_integer(arc_process.clone()),
                    strategy::term::is_not_integer(arc_process.clone()),
                ),
                |(left, right)| {
                    prop_assert_eq!(erlang::bor_2(left, right, &arc_process), Err(badarith!()));

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
                    prop_assert_eq!(erlang::bor_2(operand, operand, &arc_process), Ok(operand));

                    Ok(())
                },
            )
            .unwrap();
    });
}
