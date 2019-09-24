use proptest::arbitrary::any;
use proptest::prop_assert_eq;
use proptest::strategy::{Just, Strategy};
use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::badarg;

use crate::otp::erlang::binary_to_float_1::native;
use crate::scheduler::with_process_arc;
use crate::test::strategy;

#[test]
fn without_binary_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_not_binary(arc_process.clone()),
                |binary| {
                    prop_assert_eq!(native(&arc_process, binary), Err(badarg!().into()));

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_binary_with_integer_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &any::<isize>().prop_flat_map(|integer| {
                    strategy::term::binary::containing_bytes(
                        integer.to_string().as_bytes().to_owned(),
                        arc_process.clone(),
                    )
                }),
                |binary| {
                    prop_assert_eq!(native(&arc_process, binary), Err(badarg!().into()));

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_binary_with_f64_returns_floats() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &any::<f64>().prop_flat_map(|f| {
                    let byte_vec = format!("{:?}", f).as_bytes().to_owned();

                    (
                        Just(f),
                        strategy::term::binary::containing_bytes(byte_vec, arc_process.clone()),
                    )
                }),
                |(f, binary)| {
                    prop_assert_eq!(
                        native(&arc_process, binary),
                        Ok(arc_process.float(f).unwrap())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_binary_with_less_than_min_f64_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::binary::containing_bytes("-1797693134862315700000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.0".as_bytes().to_owned(), arc_process.clone()),
                |binary| {
                    prop_assert_eq!(
                        native(&arc_process, binary),
                        Err(badarg!().into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_binary_with_greater_than_max_f64_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::binary::containing_bytes("1797693134862315700000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.0".as_bytes().to_owned(), arc_process.clone()),
                |binary| {
                    prop_assert_eq!(
                        native(&arc_process, binary),
                        Err(badarg!().into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}
