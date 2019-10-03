use super::*;

use crate::binary_to_string::binary_to_string;
use crate::otp::erlang::binary_to_integer_1;

#[test]
fn with_small_integer_returns_binary() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::integer::small::isize(), |integer_isize| {
                let integer = arc_process.integer(integer_isize).unwrap();

                let result = native(&arc_process, integer);

                prop_assert!(result.is_ok());

                let term = result.unwrap();

                prop_assert!(term.is_binary());

                let string: String = binary_to_string(term).unwrap();

                prop_assert_eq!(string, integer_isize.to_string());

                Ok(())
            })
            .unwrap();
    });
}

#[test]
fn with_big_integer_returns_binary() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::integer::big::isize(), |integer_isize| {
                let integer = arc_process.integer(integer_isize).unwrap();

                let result = native(&arc_process, integer);

                prop_assert!(result.is_ok());

                let term = result.unwrap();

                prop_assert!(term.is_binary());

                let string: String = binary_to_string(term).unwrap();

                prop_assert_eq!(string, integer_isize.to_string());

                Ok(())
            })
            .unwrap();
    });
}

#[test]
fn dual_of_binary_to_integer_1() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_integer(arc_process.clone()),
                |integer| {
                    let result_binary = native(&arc_process, integer);

                    prop_assert!(result_binary.is_ok());

                    let binary = result_binary.unwrap();

                    prop_assert_eq!(
                        binary_to_integer_1::native(&arc_process, binary),
                        Ok(integer)
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}
