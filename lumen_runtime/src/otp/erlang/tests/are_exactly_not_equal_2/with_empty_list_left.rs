use super::*;

use proptest::strategy::Strategy;

#[test]
fn without_empty_list_returns_true() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    Just(Term::NIL),
                    strategy::term(arc_process.clone())
                        .prop_filter("Right must not be empty list", |v| !v.is_nil()),
                ),
                |(left, right)| {
                    prop_assert_eq!(erlang::are_exactly_not_equal_2(left, right), true.into());

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_empty_list_right_returns_false() {
    assert_eq!(
        erlang::are_exactly_not_equal_2(Term::NIL, Term::NIL),
        false.into()
    );
}
