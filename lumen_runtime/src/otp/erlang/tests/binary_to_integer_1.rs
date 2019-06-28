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
                        Err(badarg!())
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

                    prop_assert_eq!(term.tag(), SmallInteger);

                    prop_assert_eq!(term, integer.into_process(&arc_process));

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

                    prop_assert_eq!(term.tag(), Boxed);

                    let unboxed: &Term = term.unbox_reference();

                    prop_assert_eq!(unboxed.tag(), BigInteger);
                    prop_assert_eq!(term, integer.into_process(&arc_process));

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
                        Err(badarg!())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}
