use super::*;

use proptest::prop_oneof;
use proptest::strategy::Strategy;

#[test]
fn without_non_empty_list_or_bitstring_second_returns_firsts() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term(arc_process.clone())
                    .prop_filter("Second cannot be a list or bitstring", |second| {
                        !(second.is_list() || second.is_bitstring())
                    }),
                |second| {
                    let first = Term::NIL;

                    prop_assert_eq!(erlang::max_2(first, second), first);

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_non_empty_list_or_bitstring_second_returns_second() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &prop_oneof![
                    strategy::term::list::non_empty_maybe_improper(arc_process.clone()),
                    strategy::term::is_bitstring(arc_process)
                ],
                |second| {
                    let first = Term::NIL;

                    prop_assert_eq!(erlang::max_2(first, second), second);

                    Ok(())
                },
            )
            .unwrap();
    });
}
