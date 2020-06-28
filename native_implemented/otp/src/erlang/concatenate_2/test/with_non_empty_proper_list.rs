use super::*;

use proptest::test_runner::TestCaseResult;

use liblumen_alloc::erts::term::list;

#[test]
fn without_list_right_returns_improper_list_with_right_as_tail() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::list::non_empty_proper(arc_process.clone()),
                strategy::term::is_not_list(arc_process.clone()),
            )
        },
        |(arc_process, left, right)| {
            let result = result(&arc_process, left, right);

            prop_assert!(result.is_ok());

            let concatenated = result.unwrap();
            let concatenated_result_cons: core::result::Result<Boxed<Cons>, _> =
                concatenated.try_into();

            prop_assert!(concatenated_result_cons.is_ok());

            let concatenated_cons = concatenated_result_cons.unwrap();
            let mut concatenated_iter = concatenated_cons.into_iter();

            match left.decode().unwrap() {
                TypedTerm::Nil => {
                    prop_assert_eq!(
                        concatenated_iter.next(),
                        Some(Err(ImproperList { tail: right }))
                    );
                    prop_assert_eq!(concatenated_iter.next(), None);
                }
                TypedTerm::List(cons) => {
                    let mut left_iter = cons.into_iter();

                    loop {
                        let left_element = left_iter.next();
                        let concatenated_element = concatenated_iter.next();

                        if left_element == None {
                            prop_assert_eq!(
                                concatenated_element,
                                Some(Err(ImproperList { tail: right }))
                            );
                            prop_assert_eq!(concatenated_iter.next(), None);

                            break;
                        } else {
                            prop_assert_eq!(left_element, concatenated_element);
                        }
                    }
                }
                _ => panic!("left ({:?}) is not a list"),
            }

            Ok(())
        },
    );
}

#[test]
fn with_improper_list_right_returns_improper_list_with_right_as_tail() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::list::non_empty_proper(arc_process.clone()),
                strategy::term::list::improper(arc_process.clone()),
            )
        },
        |(arc_process, left, right)| {
            let result = result(&arc_process, left, right);

            prop_assert!(result.is_ok());

            let concatenated = result.unwrap();
            let concatenated_result_cons: core::result::Result<Boxed<Cons>, _> =
                concatenated.try_into();

            prop_assert!(concatenated_result_cons.is_ok());

            let concatenated_cons = concatenated_result_cons.unwrap();

            let mut concatenated_iter = concatenated_cons.into_iter();

            match left.decode().unwrap() {
                TypedTerm::Nil => {
                    prop_assert_eq!(
                        concatenated_iter.next(),
                        Some(Err(ImproperList { tail: right }))
                    );
                    prop_assert_eq!(concatenated_iter.next(), None);

                    Ok(())
                }
                TypedTerm::List(left_cons) => {
                    prop_assert_right_as_tail(concatenated_iter, left_cons, right)
                }
                _ => panic!("left ({:?}) is not a list"),
            }
        },
    );
}

#[test]
fn with_list_right_returns_proper_list_with_right_as_tail() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::list::non_empty_proper(arc_process.clone()),
                strategy::term::list::non_empty_proper(arc_process.clone()),
            )
        },
        |(arc_process, left, right)| {
            let result = result(&arc_process, left, right);

            prop_assert!(result.is_ok());

            let concatenated = result.unwrap();
            let concatenated_result_cons: core::result::Result<Boxed<Cons>, _> =
                concatenated.try_into();

            prop_assert!(concatenated_result_cons.is_ok());

            let concatenated_cons = concatenated_result_cons.unwrap();

            let mut concatenated_iter = concatenated_cons.into_iter();

            match left.decode().unwrap() {
                TypedTerm::Nil => {
                    let right_cons: Boxed<Cons> = right.try_into().unwrap();
                    let mut right_iter = right_cons.into_iter();

                    loop {
                        let right_element = right_iter.next();
                        let concatenated_element = concatenated_iter.next();

                        match (&right_element, &concatenated_element) {
                            (None, None) => break,
                            _ => prop_assert_eq!(right_element, concatenated_element),
                        }
                    }

                    Ok(())
                }
                TypedTerm::List(left_cons) => {
                    prop_assert_right_as_tail(concatenated_iter, left_cons, right)
                }
                _ => panic!("left ({:?}) is not a list"),
            }
        },
    );
}

fn prop_assert_right_as_tail(
    mut concatenated_iter: list::Iter,
    left_cons: Boxed<Cons>,
    right: Term,
) -> TestCaseResult {
    let mut left_iter = left_cons.into_iter();

    loop {
        let left_element = left_iter.next();

        if left_element == None {
            let right_cons: Boxed<Cons> = right.try_into().unwrap();
            let mut right_iter = right_cons.into_iter();

            loop {
                let right_element = right_iter.next();
                let concatenated_element = concatenated_iter.next();

                match (&right_element, &concatenated_element) {
                    (None, None) => break,
                    _ => prop_assert_eq!(right_element, concatenated_element),
                }
            }

            break;
        } else {
            let concatenated_element = concatenated_iter.next();

            prop_assert_eq!(left_element, concatenated_element);
        }
    }

    Ok(())
}
