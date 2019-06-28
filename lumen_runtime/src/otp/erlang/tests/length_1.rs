use super::*;

use proptest::collection::SizeRange;
use proptest::strategy::Strategy;

#[test]
fn without_list_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::is_not_list(arc_process.clone()), |list| {
                prop_assert_eq!(erlang::length_1(list, &arc_process), Err(badarg!()));

                Ok(())
            })
            .unwrap();
    });
}

#[test]
fn with_empty_list_is_zero() {
    with_process(|process| {
        let list = Term::EMPTY_LIST;
        let zero_term = 0.into_process(&process);

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
                    prop_assert_eq!(erlang::length_1(list, &arc_process), Err(badarg!()));

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
                            Term::slice_to_list(&element_vec, &arc_process),
                            element_vec.len(),
                        )
                    }),
                |(list, element_count)| {
                    prop_assert_eq!(
                        erlang::length_1(list, &arc_process),
                        Ok(element_count.into_process(&arc_process))
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}
