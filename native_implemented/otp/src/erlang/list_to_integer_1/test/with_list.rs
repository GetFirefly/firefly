use super::*;

use std::convert::TryInto;

use liblumen_alloc::erts::term::prelude::SmallInteger;

#[test]
fn with_small_integer_returns_small_integer() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::integer::small::isize(),
            )
                .prop_map(|(arc_process, integer)| {
                    (
                        arc_process.clone(),
                        integer,
                        arc_process.charlist_from_str(&integer.to_string()).unwrap(),
                    )
                })
        },
        |(arc_process, integer, list)| {
            let result = result(&arc_process, list);

            prop_assert!(result.is_ok());

            let term = result.unwrap();

            let small_integer_result: core::result::Result<SmallInteger, _> = term.try_into();

            prop_assert!(small_integer_result.is_ok());

            let small_integer = small_integer_result.unwrap();
            let small_integer_isize: isize = small_integer.into();

            prop_assert_eq!(small_integer_isize, integer);

            Ok(())
        },
    );
}

#[test]
fn with_big_integer_returns_big_integer() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::integer::big::isize(),
            )
                .prop_map(|(arc_process, integer)| {
                    (
                        arc_process.clone(),
                        integer,
                        arc_process.charlist_from_str(&integer.to_string()).unwrap(),
                    )
                })
        },
        |(arc_process, integer, list)| {
            let result = result(&arc_process, list);

            prop_assert!(result.is_ok());

            let term = result.unwrap();

            prop_assert!(term.is_boxed_bigint());
            prop_assert_eq!(term, arc_process.integer(integer).unwrap());

            Ok(())
        },
    );
}

#[test]
fn with_non_decimal_errors_badarg() {
    with_process_arc(|arc_process| {
        let string = "FF";
        let list = arc_process.charlist_from_str(&string).unwrap();

        assert_badarg!(
            result(&arc_process, list),
            format!("list ('{}') is not base 10", string)
        );
    });
}
