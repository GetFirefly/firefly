mod with_atom;
mod with_small_integer;

use proptest::strategy::Strategy;
use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::atom;
use liblumen_alloc::erts::term::prelude::*;

use crate::otp::erlang::system_time_1::native;
use crate::scheduler::{with_process, with_process_arc};
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
                    prop_assert_badarg!(native(&arc_process, unit,), "supported units are :second, :seconds, :millisecond, :milli_seconds, :microsecond, :micro_seconds, :nanosecond, :nano_seconds, :native, :perf_counter, or hertz (positive integer)");

                    Ok(())
                },
            )
            .unwrap();
    });
}
