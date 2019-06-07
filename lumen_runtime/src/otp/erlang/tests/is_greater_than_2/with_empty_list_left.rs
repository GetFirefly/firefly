use super::*;

use proptest::prop_oneof;
use proptest::strategy::Strategy;

#[test]
fn without_non_empty_list_or_bitstring_returns_true() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term(arc_process.clone())
                    .prop_filter("Right cannot be a list or bitstring", |right| {
                        !(right.is_list() || right.is_bitstring())
                    }),
                |right| {
                    let left = Term::EMPTY_LIST;

                    prop_assert_eq!(erlang::is_greater_than_2(left, right), true.into());

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_non_empty_list_or_bitstring_right_returns_false() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &prop_oneof![
                    strategy::term::list::non_empty_maybe_improper(arc_process.clone()),
                    strategy::term::is_bitstring(arc_process)
                ],
                |right| {
                    let left = Term::EMPTY_LIST;

                    prop_assert_eq!(erlang::is_greater_than_2(left, right), false.into());

                    Ok(())
                },
            )
            .unwrap();
    });
}
