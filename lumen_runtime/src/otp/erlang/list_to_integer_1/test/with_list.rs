use super::*;

use std::convert::TryInto;

use liblumen_alloc::erts::term::prelude::SmallInteger;

#[test]
fn with_small_integer_returns_small_integer() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::integer::small::isize().prop_map(|integer| {
                    (
                        integer,
                        arc_process.charlist_from_str(&integer.to_string()).unwrap(),
                    )
                }),
                |(integer, list)| {
                    let result = native(&arc_process, list);

                    prop_assert!(result.is_ok());

                    let term = result.unwrap();

                    let small_integer_result: core::result::Result<SmallInteger, _> =
                        term.try_into();

                    prop_assert!(small_integer_result.is_ok());

                    let small_integer = small_integer_result.unwrap();
                    let small_integer_isize: isize = small_integer.into();

                    prop_assert_eq!(small_integer_isize, integer);

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_big_integer_returns_big_integer() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::integer::big::isize().prop_map(|integer| {
                    (
                        integer,
                        arc_process.charlist_from_str(&integer.to_string()).unwrap(),
                    )
                }),
                |(integer, list)| {
                    let result = native(&arc_process, list);

                    prop_assert!(result.is_ok());

                    let term = result.unwrap();

                    prop_assert!(term.is_boxed_bigint());
                    prop_assert_eq!(term, arc_process.integer(integer).unwrap());

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_non_decimal_errors_badarg() {
    with_process_arc(|arc_process| {
        let string = "FF";
        let list = arc_process.charlist_from_str(&string).unwrap();

        assert_badarg!(
            native(&arc_process, list),
            format!("string ({}) is not base 10", string)
        );
    });
}
