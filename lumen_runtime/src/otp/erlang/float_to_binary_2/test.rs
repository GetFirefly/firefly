mod with_float;

use proptest::test_runner::{Config, TestRunner};
use proptest::{prop_assert, prop_assert_eq};

use liblumen_alloc::atom;
use liblumen_alloc::erts::term::prelude::*;

use crate::otp::erlang::float_to_binary_2::native;
use crate::scheduler::with_process_arc;
use crate::test::strategy;

#[test]
fn without_float_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_not_float(arc_process.clone()),
                |float| {
                    let options = Term::NIL;

                    prop_assert_badarg!(
                        native(&arc_process, float, options),
                        format!("float ({}) is not a float", float)
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}
