mod with_map;

use proptest::prop_assert_eq;
use proptest::strategy::Strategy;
use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::badmap;

use crate::otp::maps::values_1::native;
use crate::scheduler::with_process_arc;
use crate::test::strategy;

#[test]
fn without_map_errors_badmap() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(strategy::term::is_not_map(arc_process.clone())),
                |non_map| {
                    prop_assert_eq!(
                        native(&arc_process, non_map),
                        Err(badmap!(&arc_process, non_map))
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}
