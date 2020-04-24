mod with_atom;
mod with_small_integer;

use proptest::strategy::{Just, Strategy};

use liblumen_alloc::atom;
use liblumen_alloc::erts::term::prelude::*;

use crate::erlang::system_time_1::result;
use crate::test::{strategy, with_process};

#[test]
fn without_atom_or_integer_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term(arc_process.clone())
                    .prop_filter("Unit must not be an atom or integer", |unit| {
                        !(unit.is_integer() || unit.is_atom())
                    }),
            )
        },
        |(arc_process, unit)| {
            prop_assert_badarg!(result(&arc_process, unit,), "supported units are :second, :seconds, :millisecond, :milli_seconds, :microsecond, :micro_seconds, :nanosecond, :nano_seconds, :native, :perf_counter, or hertz (positive integer)");

            Ok(())
        },
    );
}
