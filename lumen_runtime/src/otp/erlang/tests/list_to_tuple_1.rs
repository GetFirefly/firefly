use super::*;

use proptest::collection::SizeRange;
use proptest::strategy::Strategy;

#[test]
fn without_list_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::is_not_list(arc_process.clone()), |list| {
                prop_assert_eq!(
                    erlang::list_to_tuple_1(list, &arc_process),
                    Err(badarg!().into())
                );

                Ok(())
            })
            .unwrap();
    });
}

#[test]
fn with_empty_list_returns_empty_tuple() {
    with_process(|process| {
        let list = Term::NIL;

        assert_eq!(
            erlang::list_to_tuple_1(list, &process),
            Ok(process.tuple_from_slice(&[]).unwrap())
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
                        let list = arc_process.list_from_slice(&vec).unwrap();
                        let tuple = arc_process.tuple_from_slice(&vec).unwrap();

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
                    prop_assert_eq!(
                        erlang::list_to_tuple_1(list, &arc_process),
                        Err(badarg!().into())
                    );

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
        let first_element = atom_unchecked("share");

        let (second_element, list) = {
            let second_element = process
                .cons(
                    atom_unchecked("Ericsson_B"),
                    process
                        .cons(process.integer(163).unwrap(), Term::NIL)
                        .unwrap(),
                )
                .unwrap();

            let list = process
                .cons(
                    first_element,
                    process.cons(second_element, Term::NIL).unwrap(),
                )
                .unwrap();

            (second_element, list)
        };

        assert_eq!(
            erlang::list_to_tuple_1(list, &process),
            Ok(process
                .tuple_from_slice(&[first_element, second_element],)
                .unwrap())
        );
    });
}
