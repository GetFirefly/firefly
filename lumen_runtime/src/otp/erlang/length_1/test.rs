use proptest::collection::SizeRange;
use proptest::prop_assert_eq;
use proptest::strategy::Strategy;
use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::erts::term::prelude::Term;

use crate::otp::erlang::length_1::native;
use crate::scheduler::{with_process, with_process_arc};
use crate::test::strategy;

#[test]
fn without_list_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::is_not_list(arc_process.clone()), |list| {
                prop_assert_badarg!(
                    native(&arc_process, list),
                    format!("list ({}) is not a list", list)
                );

                Ok(())
            })
            .unwrap();
    });
}

#[test]
fn with_empty_list_is_zero() {
    with_process(|process| {
        let list = Term::NIL;
        let zero_term = process.integer(0).unwrap();

        assert_eq!(native(process, list), Ok(zero_term));
    });
}

#[test]
fn with_improper_list_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::list::improper(arc_process.clone()),
                |list| {
                    prop_assert_badarg!(
                        native(&arc_process, list),
                        format!("list ({}) is improper", list)
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_non_empty_proper_list_is_number_of_elements() {
    with_process_arc(|arc_process| {
        let size_range: SizeRange = strategy::NON_EMPTY_RANGE_INCLUSIVE.clone().into();

        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(proptest::collection::vec(strategy::term(arc_process.clone()), size_range))
                    .prop_map(|element_vec| {
                        (
                            arc_process.list_from_slice(&element_vec).unwrap(),
                            element_vec.len(),
                        )
                    }),
                |(list, element_count)| {
                    prop_assert_eq!(
                        native(&arc_process, list),
                        Ok(arc_process.integer(element_count).unwrap())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}
