use super::*;

use proptest::strategy::Strategy;

#[test]
fn without_binary_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_not_binary(arc_process.clone()),
                |binary| {
                    prop_assert_eq!(
                        erlang::binary_to_integer_1(binary, &arc_process),
                        Err(badarg!().into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_binary_with_small_integer_returns_small_integer() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::integer::small::isize().prop_flat_map(|integer| {
                    let byte_vec = integer.to_string().as_bytes().to_owned();

                    (
                        Just(integer),
                        strategy::term::binary::containing_bytes(byte_vec, arc_process.clone()),
                    )
                }),
                |(integer, binary)| {
                    let result = erlang::binary_to_integer_1(binary, &arc_process);

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
fn with_binary_with_big_integer_returns_big_integer() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::integer::big::isize().prop_flat_map(|integer| {
                    let byte_vec = integer.to_string().as_bytes().to_owned();

                    (
                        Just(integer),
                        strategy::term::binary::containing_bytes(byte_vec, arc_process.clone()),
                    )
                }),
                |(integer, binary)| {
                    let result = erlang::binary_to_integer_1(binary, &arc_process);

                    prop_assert!(result.is_ok());

                    let term = result.unwrap();

                    prop_assert!(term.is_bigint());
                    prop_assert_eq!(term, arc_process.integer(integer));

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_binary_with_non_decimal_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::binary::containing_bytes(
                    "FF".as_bytes().to_owned(),
                    arc_process.clone(),
                ),
                |binary| {
                    prop_assert_eq!(
                        erlang::binary_to_integer_1(binary, &arc_process),
                        Err(badarg!().into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}
