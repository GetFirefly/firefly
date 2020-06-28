use proptest::strategy::{Just, Strategy};

use liblumen_alloc::atom;
use liblumen_alloc::erts::term::prelude::*;

use crate::erlang::unique_integer_1::result;
use crate::test::strategy;
use crate::test::with_process;

#[test]
fn without_proper_list_of_options_errors_badargs() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term(arc_process.clone()).prop_filter(
                    "Cannot be a proper list of valid options",
                    |term| match term.decode().unwrap() {
                        TypedTerm::Nil => false,
                        TypedTerm::List(cons) => {
                            let mut filter = true;

                            for result in cons.into_iter() {
                                match result {
                                    Ok(element) => match element.decode().unwrap() {
                                        TypedTerm::Atom(atom) => match atom.name() {
                                            "monotonic" | "positive" => {
                                                filter = false;

                                                break;
                                            }
                                            _ => break,
                                        },
                                        _ => continue,
                                    },
                                    Err(_) => break,
                                }
                            }

                            filter
                        }
                        _ => true,
                    },
                ),
            )
        },
        |(arc_process, options)| {
            prop_assert_badarg!(result(&arc_process, options), "improper list");

            Ok(())
        },
    );
}

#[test]
fn without_options_returns_non_monotonic_negative_and_positive_integer() {
    const OPTIONS: Term = Term::NIL;

    with_process(|process| {
        let result_first_unique_integer = result(process, OPTIONS);

        assert!(result_first_unique_integer.is_ok());

        let first_unique_integer = result_first_unique_integer.unwrap();
        let zero = process.integer(0).unwrap();

        assert!(first_unique_integer.is_integer());
        assert!(first_unique_integer <= zero);

        let result_second_unique_integer = result(process, OPTIONS);

        assert!(result_second_unique_integer.is_ok());

        let second_unique_integer = result_second_unique_integer.unwrap();

        assert!(second_unique_integer.is_integer());
        assert!(second_unique_integer <= zero);

        assert_ne!(first_unique_integer, second_unique_integer);
    });
}

#[test]
fn with_monotonic_returns_monotonic_negative_and_positiver_integer() {
    with_process(|process| {
        let options = process.list_from_slice(&[atom!("monotonic")]).unwrap();

        let result_first_unique_integer = result(process, options);

        assert!(result_first_unique_integer.is_ok());

        let first_unique_integer = result_first_unique_integer.unwrap();
        let zero = process.integer(0).unwrap();

        assert!(first_unique_integer.is_integer());
        assert!(first_unique_integer <= zero);

        let result_second_unique_integer = result(process, options);

        assert!(result_second_unique_integer.is_ok());

        let second_unique_integer = result_second_unique_integer.unwrap();

        assert!(second_unique_integer.is_integer());
        assert!(second_unique_integer <= zero);

        assert!(first_unique_integer < second_unique_integer);
    });
}

#[test]
fn with_monotonic_and_positive_returns_monotonic_positiver_integer() {
    with_process(|process| {
        let options = process
            .list_from_slice(&[atom!("monotonic"), atom!("positive")])
            .unwrap();

        let result_first_unique_integer = result(process, options);

        assert!(result_first_unique_integer.is_ok());

        let first_unique_integer = result_first_unique_integer.unwrap();
        let zero = process.integer(0).unwrap();

        assert!(first_unique_integer.is_integer());
        assert!(zero <= first_unique_integer);

        let result_second_unique_integer = result(process, options);

        assert!(result_second_unique_integer.is_ok());

        let second_unique_integer = result_second_unique_integer.unwrap();

        assert!(second_unique_integer.is_integer());
        assert!(zero <= second_unique_integer);

        assert!(first_unique_integer < second_unique_integer);
    });
}

#[test]
fn with_positive_returns_non_monotonic_and_positive_integer() {
    with_process(|process| {
        let options = process.list_from_slice(&[atom!("positive")]).unwrap();

        let result_first_unique_integer = result(process, options);

        assert!(result_first_unique_integer.is_ok());

        let first_unique_integer = result_first_unique_integer.unwrap();
        let zero = process.integer(0).unwrap();

        assert!(first_unique_integer.is_integer());
        assert!(zero <= first_unique_integer);

        let result_second_unique_integer = result(process, options);

        assert!(result_second_unique_integer.is_ok());

        let second_unique_integer = result_second_unique_integer.unwrap();

        assert!(second_unique_integer.is_integer());
        assert!(zero <= second_unique_integer);

        assert_ne!(first_unique_integer, second_unique_integer);
    });
}
