use super::*;

use proptest::strategy::Strategy;

#[test]
fn without_empty_list_returns_false() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    Just(Term::NIL),
                    strategy::term(arc_process.clone())
                        .prop_filter("Right must not be empty list", |v| !v.is_nil()),
                ),
                |(left, right)| {
                    prop_assert_eq!(erlang::are_exactly_equal_2(left, right), false.into());

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_empty_list_right_returns_true() {
    assert_eq!(
        erlang::are_exactly_equal_2(Term::NIL, Term::NIL),
        true.into()
    );
}
