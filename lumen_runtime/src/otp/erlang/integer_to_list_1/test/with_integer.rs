use super::*;

use std::convert::TryInto;

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

                let string: String = term.try_into().unwrap();

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

                let string: String = term.try_into().unwrap();

                prop_assert_eq!(string, integer_isize.to_string());

                Ok(())
            })
            .unwrap();
    });
}
