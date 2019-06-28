use super::*;

mod with_big_integer_augend;
mod with_float_augend;
mod with_small_integer_augend;

#[test]
fn without_number_augend_errors_badarith() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_not_number(arc_process.clone()),
                    strategy::term::is_number(arc_process.clone()),
                ),
                |(augend, addend)| {
                    prop_assert_eq!(
                        erlang::add_2(augend, addend, &arc_process),
                        Err(badarith!())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}
