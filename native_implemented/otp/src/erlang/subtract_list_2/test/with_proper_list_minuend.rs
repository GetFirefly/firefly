use super::*;

#[test]
fn without_proper_list_subtrahend_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::list::proper(arc_process.clone()),
                strategy::term::list::improper(arc_process.clone()),
            )
        },
        |(arc_process, minuend, subtrahend)| {
            prop_assert_badarg!(
                result(&arc_process, minuend, subtrahend),
                format!("subtrahend ({}) is not a proper list", subtrahend)
            );

            Ok(())
        },
    );
}

#[test]
fn with_subtrahend_list_returns_minuend_with_first_copy_of_each_element_in_subtrahend_removed() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term(arc_process.clone()),
                strategy::term(arc_process.clone()),
            )
        },
        |(arc_process, element1, element2)| {
            let minuend = arc_process
                .list_from_slice(&[element1, element2, element1])
                .unwrap();
            let subtrahend = arc_process.list_from_slice(&[element1]).unwrap();

            prop_assert_eq!(
                result(&arc_process, minuend, subtrahend),
                Ok(arc_process.list_from_slice(&[element2, element1]).unwrap())
            );

            Ok(())
        },
    );
}
