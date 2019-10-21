use proptest::prop_assert_eq;
use proptest::strategy::Strategy;
use proptest::test_runner::{Config, TestRunner};

use crate::otp::erlang::monotonic_time_1;
use crate::otp::erlang::subtract_2;
use crate::otp::erlang::system_time_1;
use crate::otp::erlang::time_offset_1;
use crate::scheduler::with_process_arc;
use crate::test::strategy;

#[test]
fn without_atom_or_integer_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term(arc_process.clone())
                    .prop_filter("Unit must not be an atom or integer", |unit| {
                        !(unit.is_integer() || unit.is_atom())
                    }),
                |unit| {
                    let monotonic_time = monotonic_time_1::native(&arc_process, unit).unwrap();
                    let system_time = system_time_1::native(&arc_process, unit).unwrap();
                    let time_offset = time_offset_1::native(&arc_process, unit).unwrap();
                    let expected_time_offset =
                        subtract_2::native(&arc_process, system_time, monotonic_time).unwrap();
                    prop_assert_eq!(time_offset, expected_time_offset);

                    Ok(())
                },
            )
            .unwrap();
    });
}
