use super::*;

use proptest::arbitrary::any;

use crate::erlang::binary_to_integer_1;
use crate::runtime::binary_to_string::binary_to_string;

#[test]
fn with_integer_returns_binary() {
    run!(
        |arc_process| { (Just(arc_process.clone()), any::<isize>()) },
        |(arc_process, integer_isize)| {
            let integer = arc_process.integer(integer_isize).unwrap();

            let result = native(&arc_process, integer);

            prop_assert!(result.is_ok());

            let term = result.unwrap();

            prop_assert!(term.is_binary());

            let string: String = binary_to_string(term).unwrap();

            prop_assert_eq!(string, integer_isize.to_string());

            Ok(())
        },
    );
}

#[test]
fn dual_of_binary_to_integer_1() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_integer(arc_process.clone()),
            )
        },
        |(arc_process, integer)| {
            let result_binary = native(&arc_process, integer);

            prop_assert!(result_binary.is_ok());

            let binary = result_binary.unwrap();

            prop_assert_eq!(
                binary_to_integer_1::native(&arc_process, binary),
                Ok(integer)
            );

            Ok(())
        },
    );
}
