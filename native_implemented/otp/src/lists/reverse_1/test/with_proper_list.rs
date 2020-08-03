use super::*;

use proptest::collection::SizeRange;

use crate::test::strategy::NON_EMPTY_RANGE_INCLUSIVE;

#[test]
fn with_empty_list_returns_empty_list() {
    with_process_arc(|arc_process| {
        assert_eq!(result(&arc_process, Term::NIL), Ok(Term::NIL));
    });
}

#[test]
fn reverses_order_of_elements_of_list() {
    let size_range: SizeRange = NON_EMPTY_RANGE_INCLUSIVE.clone().into();

    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &proptest::collection::vec(strategy::term(arc_process.clone()), size_range),
                |vec| {
                    let list = arc_process.list_from_slice(&vec).unwrap();

                    let reversed_vec: Vec<Term> = vec.iter().rev().copied().collect();
                    let reversed = arc_process.list_from_slice(&reversed_vec).unwrap();

                    prop_assert_eq!(result(&arc_process, list), Ok(reversed));

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn reverse_reverse_returns_original_list() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::list::proper(arc_process.clone()), |list| {
                let reversed = result(&arc_process, list).unwrap();

                prop_assert_eq!(result(&arc_process, reversed), Ok(list));

                Ok(())
            })
            .unwrap();
    });
}
