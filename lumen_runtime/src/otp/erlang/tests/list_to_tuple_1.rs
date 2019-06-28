use super::*;

use proptest::collection::SizeRange;
use proptest::strategy::Strategy;

#[test]
fn without_list_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::is_not_list(arc_process.clone()), |list| {
                prop_assert_eq!(erlang::list_to_tuple_1(list, &arc_process), Err(badarg!()));

                Ok(())
            })
            .unwrap();
    });
}

#[test]
fn with_empty_list_returns_empty_tuple() {
    with_process(|process| {
        let list = Term::EMPTY_LIST;

        assert_eq!(
            erlang::list_to_tuple_1(list, &process),
            Ok(Term::slice_to_tuple(&[], &process))
        );
    });
}

#[test]
fn with_non_empty_proper_list_returns_tuple() {
    with_process_arc(|arc_process| {
        let size_range: SizeRange = strategy::NON_EMPTY_RANGE_INCLUSIVE.clone().into();

        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &proptest::collection::vec(strategy::term(arc_process.clone()), size_range)
                    .prop_map(|vec| {
                        let list = Term::slice_to_list(&vec, &arc_process);
                        let tuple = Term::slice_to_tuple(&vec, &arc_process);

                        (list, tuple)
                    }),
                |(list, tuple)| {
                    prop_assert_eq!(erlang::list_to_tuple_1(list, &arc_process), Ok(tuple));

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_improper_list_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::list::improper(arc_process.clone()),
                |list| {
                    prop_assert_eq!(erlang::list_to_tuple_1(list, &arc_process), Err(badarg!()));

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_nested_list_returns_tuple_with_list_element() {
    with_process(|process| {
        // erlang doc: `[share, ['Ericsson_B', 163]]`
        let first_element = Term::str_to_atom("share", DoNotCare).unwrap();
        let second_element = Term::cons(
            Term::str_to_atom("Ericsson_B", DoNotCare).unwrap(),
            Term::cons(163.into_process(&process), Term::EMPTY_LIST, &process),
            &process,
        );
        let list = Term::cons(
            first_element,
            Term::cons(second_element, Term::EMPTY_LIST, &process),
            &process,
        );

        assert_eq!(
            erlang::list_to_tuple_1(list, &process),
            Ok(Term::slice_to_tuple(
                &[first_element, second_element],
                &process
            ))
        );
    });
}
