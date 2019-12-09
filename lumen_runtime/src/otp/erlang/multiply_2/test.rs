mod with_big_integer_multiplier;
mod with_float_multiplier;
mod with_small_integer_multiplier;

use proptest::prop_assert;
use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::otp::erlang::multiply_2::native;
use crate::scheduler::{with_process, with_process_arc};
use crate::test::strategy;

#[test]
fn without_number_multiplier_errors_badarith() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_not_number(arc_process.clone()),
                    strategy::term::is_number(arc_process.clone()),
                ),
                |(multiplier, multiplicand)| {
                    prop_assert_badarith!(
                        native(&arc_process, multiplier, multiplicand),
                        format!(
                            "multiplier ({}) and multiplicand ({}) aren't both numbers",
                            multiplier, multiplicand
                        )
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}
