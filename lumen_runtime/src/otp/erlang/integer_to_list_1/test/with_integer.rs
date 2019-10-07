use super::*;

use crate::otp::erlang::list_to_integer_1;
use crate::otp::erlang::list_to_string::list_to_string;

#[test]
fn with_small_integer_returns_list() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::integer::small::isize(), |integer_isize| {
                let integer = arc_process.integer(integer_isize).unwrap();

                let result = native(&arc_process, integer);

                prop_assert!(result.is_ok());

                let term = result.unwrap();

                prop_assert!(term.is_list());

                let string: String = list_to_string(term).unwrap();

                prop_assert_eq!(string, integer_isize.to_string());

                Ok(())
            })
            .unwrap();
    });
}

#[test]
fn with_big_integer_returns_list() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::integer::big::isize(), |integer_isize| {
                let integer = arc_process.integer(integer_isize).unwrap();

                let result = native(&arc_process, integer);

                prop_assert!(result.is_ok());

                let term = result.unwrap();

                prop_assert!(term.is_list());

                let string: String = list_to_string(term).unwrap();

                prop_assert_eq!(string, integer_isize.to_string());

                Ok(())
            })
            .unwrap();
    });
}

#[test]
fn is_dual_of_list_to_integer_1() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_integer(arc_process.clone()),
                |integer| {
                    let result_list = native(&arc_process, integer);

                    prop_assert!(result_list.is_ok());

                    let list = result_list.unwrap();

                    prop_assert_eq!(list_to_integer_1::native(&arc_process, list), Ok(integer));

                    Ok(())
                },
            )
            .unwrap();
    });
}
