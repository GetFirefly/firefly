use super::*;

use proptest::collection::SizeRange;

use crate::test::strategy::NON_EMPTY_RANGE_INCLUSIVE;

#[test]
fn with_empty_list_returns_tail() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term(arc_process.clone()), |tail| {
                prop_assert_eq!(result(&arc_process, Term::NIL, tail), Ok(tail));

                Ok(())
            })
            .unwrap();
    });
}

#[test]
fn reverses_order_of_elements_of_list_and_concatenate_tail() {
    let size_range: SizeRange = NON_EMPTY_RANGE_INCLUSIVE.clone().into();

    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    proptest::collection::vec(strategy::term(arc_process.clone()), size_range),
                    strategy::term(arc_process.clone()),
                ),
                |(vec, tail)| {
                    let list = arc_process.list_from_slice(&vec).unwrap();

                    let reversed_vec: Vec<Term> = vec.iter().rev().copied().collect();
                    let reversed_with_tail = arc_process
                        .improper_list_from_slice(&reversed_vec, tail)
                        .unwrap();

                    prop_assert_eq!(result(&arc_process, list, tail), Ok(reversed_with_tail));

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn reverse_reverse_with_empty_list_tail_returns_original_list() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::list::proper(arc_process.clone()), |list| {
                let tail = Term::NIL;
                let reversed_with_tail = result(&arc_process, list, tail).unwrap();

                prop_assert_eq!(result(&arc_process, reversed_with_tail, tail), Ok(list));

                Ok(())
            })
            .unwrap();
    });
}
