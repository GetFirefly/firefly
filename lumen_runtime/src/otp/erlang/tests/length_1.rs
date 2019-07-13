use super::*;

use proptest::collection::SizeRange;
use proptest::strategy::Strategy;

#[test]
fn without_list_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::is_not_list(arc_process.clone()), |list| {
                prop_assert_eq!(erlang::length_1(list, &arc_process), Err(badarg!().into()));

                Ok(())
            })
            .unwrap();
    });
}

#[test]
fn with_empty_list_is_zero() {
    with_process(|process| {
        let list = Term::NIL;
        let zero_term = process.integer(0);

        assert_eq!(erlang::length_1(list, &process), Ok(zero_term));
    });
}

#[test]
fn with_improper_list_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::list::improper(arc_process.clone()),
                |list| {
                    prop_assert_eq!(erlang::length_1(list, &arc_process), Err(badarg!().into()));

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
                        erlang::length_1(list, &arc_process),
                        Ok(arc_process.integer(element_count))
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}
