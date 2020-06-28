use proptest::collection::SizeRange;
use proptest::prop_assert_eq;
use proptest::strategy::{Just, Strategy};
use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::erts::term::prelude::*;

use crate::erlang::list_to_tuple_1::result;
use crate::test::strategy;
use crate::test::{with_process, with_process_arc};

#[test]
fn without_list_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::is_not_list(arc_process.clone()), |list| {
                prop_assert_badarg!(
                    result(&arc_process, list),
                    format!("list ({}) is not a list", list)
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
            result(process, list),
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
                    prop_assert_eq!(result(&arc_process, list), Ok(tuple));

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_improper_list_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::list::improper(arc_process.clone()),
            )
        },
        |(arc_process, list)| {
            prop_assert_badarg!(
                result(&arc_process, list),
                format!("list ({}) is improper", list)
            );

            Ok(())
        },
    );
}

#[test]
fn with_nested_list_returns_tuple_with_list_element() {
    with_process(|process| {
        // erlang doc: `[share, ['Ericsson_B', 163]]`
        let first_element = Atom::str_to_term("share");

        let (second_element, list) = {
            let second_element = process
                .cons(
                    Atom::str_to_term("Ericsson_B"),
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
            result(process, list),
            Ok(process
                .tuple_from_slice(&[first_element, second_element],)
                .unwrap())
        );
    });
}
