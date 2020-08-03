use super::*;

use crate::erlang::list_to_string::list_to_string;
use proptest::arbitrary::any;
use proptest::strategy::{Just, Strategy};

#[test]
fn without_base_base_errors_badarg() {
    crate::test::with_integer_integer_without_base_base_errors_badarg(file!(), result);
}

#[test]
fn with_base_base_returns_list() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                any::<isize>(),
                strategy::base::base(),
            )
        },
        |(arc_process, integer_isize, base_u8)| {
            let integer = arc_process.integer(integer_isize).unwrap();
            let base = arc_process.integer(base_u8).unwrap();

            let result = result(&arc_process, integer, base);

            prop_assert!(result.is_ok());

            let list = result.unwrap();

            prop_assert!(list.is_list());

            Ok(())
        },
    );
}

#[test]
fn with_negative_integer_returns_list_in_base_with_negative_sign_in_front_of_non_negative_list() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                (std::isize::MIN..=-1_isize),
                strategy::base::base(),
            )
                .prop_flat_map(|(arc_process, negative_isize, base_u8)| {
                    (
                        Just(arc_process.clone()),
                        Just(negative_isize),
                        Just(base_u8),
                    )
                })
        },
        |(arc_process, negative_isize, base_u8)| {
            let base = arc_process.integer(base_u8).unwrap();

            let positive_isize = -1 * negative_isize;
            let positive_integer = arc_process.integer(positive_isize).unwrap();
            let positive_list = result(&arc_process, positive_integer, base).unwrap();
            let positive_string: String = list_to_string(positive_list).unwrap();
            let expected_negative_string = format!("-{}", positive_string);
            let expected_negative_list = arc_process
                .charlist_from_str(&expected_negative_string)
                .unwrap();

            let negative_integer = arc_process.integer(negative_isize).unwrap();

            let result = result(&arc_process, negative_integer, base);

            prop_assert!(result.is_ok());

            let list = result.unwrap();

            prop_assert_eq!(list, expected_negative_list);

            Ok(())
        },
    );
}
