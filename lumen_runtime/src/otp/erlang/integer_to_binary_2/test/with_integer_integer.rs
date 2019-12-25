use super::*;

use proptest::arbitrary::any;

use crate::binary_to_string::binary_to_string;
use crate::otp::erlang::binary_to_integer_2;

#[test]
fn without_base_base_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_integer(arc_process.clone()),
                    strategy::term::is_not_base(arc_process.clone()),
                ),
                |(integer, base)| {
                    prop_assert_badarg!(
                        native(&arc_process, integer, base),
                        "base must be an integer in 2-36"
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_base_base_returns_binary() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(any::<isize>(), strategy::base::base()),
                |(integer_isize, base_u8)| {
                    let integer = arc_process.integer(integer_isize).unwrap();
                    let base = arc_process.integer(base_u8).unwrap();

                    let result = native(&arc_process, integer, base);

                    prop_assert!(result.is_ok());

                    let binary = result.unwrap();

                    prop_assert!(binary.is_binary());

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_negative_integer_returns_binary_in_base_with_negative_sign_in_front_of_non_negative_binary()
{
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &((std::isize::MIN..=-1_isize), strategy::base::base()),
                |(negative_isize, base_u8)| {
                    let base = arc_process.integer(base_u8).unwrap();

                    let positive_isize = -1 * negative_isize;
                    let positive_integer = arc_process.integer(positive_isize).unwrap();
                    let positive_binary = native(&arc_process, positive_integer, base).unwrap();
                    let positive_string: String = binary_to_string(positive_binary).unwrap();
                    let expected_negative_string = format!("-{}", positive_string);
                    let expected_negative_binary = arc_process
                        .binary_from_str(&expected_negative_string)
                        .unwrap();

                    let negative_integer = arc_process.integer(negative_isize).unwrap();

                    let result = native(&arc_process, negative_integer, base);

                    prop_assert!(result.is_ok());

                    let binary = result.unwrap();

                    prop_assert_eq!(binary, expected_negative_binary);

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn is_dual_of_binary_to_integer_2() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_integer(arc_process.clone()),
                    strategy::term::is_base(arc_process.clone()),
                ),
                |(integer, base)| {
                    let result = native(&arc_process, integer, base);

                    prop_assert!(result.is_ok());

                    let binary = result.unwrap();
                    let binary_integer = binary_to_integer_2::native(&arc_process, binary, base).unwrap();

                    prop_assert_eq!(
                        binary_integer,
                        integer,
                        ":erlang.integer_to_binary({}, {}) # {}\n:erlang.binary_to_integer({}, {}) # {}",
                        integer,
                        base,
                        binary,
                        binary,
                        base,
                        binary_integer
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}
