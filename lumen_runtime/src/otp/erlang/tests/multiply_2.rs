use super::*;

mod with_big_integer_multiplier;
mod with_float_multiplier;
mod with_small_integer_multiplier;

#[test]
fn without_number_multiplier_errors_badarith() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_not_number(arc_process.clone()),
                    strategy::term::is_number(arc_process.clone()),
                ),
                |(multiplier, multiplicand)| {
                    prop_assert_eq!(
                        erlang::multiply_2(multiplier, multiplicand, &arc_process),
                        Err(badarith!())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}
