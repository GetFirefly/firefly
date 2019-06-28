use super::*;

use crate::list::ImproperList;

#[test]
fn without_list_right_returns_improper_list_with_right_as_tail() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::list::non_empty_proper(arc_process.clone()),
                    strategy::term::is_not_list(arc_process.clone()),
                ),
                |(left, right)| {
                    let result = erlang::concatenate_2(left, right, &arc_process);

                    prop_assert!(result.is_ok());

                    let concatenated = result.unwrap();

                    prop_assert_eq!(concatenated.tag(), List);

                    let concatenated_cons: &Cons = unsafe { concatenated.as_ref_cons_unchecked() };

                    let mut concatenated_iter = concatenated_cons.into_iter();

                    match left.tag() {
                        EmptyList => {
                            prop_assert_eq!(
                                concatenated_iter.next(),
                                Some(Err(ImproperList { tail: right }))
                            );
                            prop_assert_eq!(concatenated_iter.next(), None);
                        }
                        List => {
                            let mut left_iter = unsafe { left.as_ref_cons_unchecked() }.into_iter();

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
            )
            .unwrap();
    });
}

#[test]
fn with_improper_list_right_returns_improper_list_with_right_as_tail() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::list::non_empty_proper(arc_process.clone()),
                    strategy::term::list::improper(arc_process.clone()),
                ),
                |(left, right)| {
                    let result = erlang::concatenate_2(left, right, &arc_process);

                    prop_assert!(result.is_ok());

                    let concatenated = result.unwrap();

                    prop_assert_eq!(concatenated.tag(), List);

                    let concatenated_cons: &Cons = unsafe { concatenated.as_ref_cons_unchecked() };

                    let mut concatenated_iter = concatenated_cons.into_iter();

                    match left.tag() {
                        EmptyList => {
                            prop_assert_eq!(
                                concatenated_iter.next(),
                                Some(Err(ImproperList { tail: right }))
                            );
                            prop_assert_eq!(concatenated_iter.next(), None);
                        }
                        List => {
                            let mut left_iter = unsafe { left.as_ref_cons_unchecked() }.into_iter();

                            loop {
                                let left_element = left_iter.next();

                                if left_element == None {
                                    let mut right_iter =
                                        unsafe { right.as_ref_cons_unchecked() }.into_iter();

                                    loop {
                                        let right_element = right_iter.next();
                                        let concatenated_element = concatenated_iter.next();

                                        match (&right_element, &concatenated_element) {
                                            (None, None) => break,
                                            _ => {
                                                prop_assert_eq!(right_element, concatenated_element)
                                            }
                                        }
                                    }

                                    break;
                                } else {
                                    let concatenated_element = concatenated_iter.next();

                                    prop_assert_eq!(left_element, concatenated_element);
                                }
                            }
                        }
                        _ => panic!("left ({:?}) is not a list"),
                    }

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_list_right_returns_proper_list_with_right_as_tail() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::list::non_empty_proper(arc_process.clone()),
                    strategy::term::list::non_empty_proper(arc_process.clone()),
                ),
                |(left, right)| {
                    let result = erlang::concatenate_2(left, right, &arc_process);

                    prop_assert!(result.is_ok());

                    let concatenated = result.unwrap();

                    prop_assert_eq!(concatenated.tag(), List);

                    let concatenated_cons: &Cons = unsafe { concatenated.as_ref_cons_unchecked() };

                    let mut concatenated_iter = concatenated_cons.into_iter();

                    match left.tag() {
                        EmptyList => {
                            let mut right_iter =
                                unsafe { right.as_ref_cons_unchecked() }.into_iter();

                            loop {
                                let right_element = right_iter.next();
                                let concatenated_element = concatenated_iter.next();

                                match (&right_element, &concatenated_element) {
                                    (None, None) => break,
                                    _ => prop_assert_eq!(right_element, concatenated_element),
                                }
                            }
                        }
                        List => {
                            let mut left_iter = unsafe { left.as_ref_cons_unchecked() }.into_iter();

                            loop {
                                let left_element = left_iter.next();

                                if left_element == None {
                                    let mut right_iter =
                                        unsafe { right.as_ref_cons_unchecked() }.into_iter();

                                    loop {
                                        let right_element = right_iter.next();
                                        let concatenated_element = concatenated_iter.next();

                                        match (&right_element, &concatenated_element) {
                                            (None, None) => break,
                                            _ => {
                                                prop_assert_eq!(right_element, concatenated_element)
                                            }
                                        }
                                    }

                                    break;
                                } else {
                                    let concatenated_element = concatenated_iter.next();

                                    prop_assert_eq!(left_element, concatenated_element);
                                }
                            }
                        }
                        _ => panic!("left ({:?}) is not a list"),
                    }

                    Ok(())
                },
            )
            .unwrap();
    });
}
