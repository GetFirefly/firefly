use super::*;

use proptest::arbitrary::any;

use crate::erlang::list_to_integer_1;
use crate::erlang::list_to_string::list_to_string;

#[test]
fn with_integer_returns_list() {
    run!(
        |arc_process| { (Just(arc_process.clone()), any::<isize>(),) },
        |(arc_process, integer_isize)| {
            let integer = arc_process.integer(integer_isize).unwrap();

            let result = result(&arc_process, integer);

            prop_assert!(result.is_ok());

            let term = result.unwrap();

            prop_assert!(term.is_list());

            let string: String = list_to_string(term).unwrap();

            prop_assert_eq!(string, integer_isize.to_string());

            Ok(())
        },
    );
}

#[test]
fn is_dual_of_list_to_integer_1() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_integer(arc_process.clone()),
            )
        },
        |(arc_process, integer)| {
            let result_list = result(&arc_process, integer);

            prop_assert!(result_list.is_ok());

            let list = result_list.unwrap();

            prop_assert_eq!(list_to_integer_1::result(&arc_process, list), Ok(integer));

            Ok(())
        },
    );
}
