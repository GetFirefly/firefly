mod with_map;

use proptest::prop_assert_eq;
use proptest::strategy::Strategy;
use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::atom;

use crate::otp::maps::find_2::native;
use crate::scheduler::with_process_arc;
use crate::test::strategy;

#[test]
fn without_map_errors_badmap() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term(arc_process.clone()),
                    strategy::term::is_not_map(arc_process.clone()),
                ),
                |(key, map)| {
                    prop_assert_badmap!(
                        native(&arc_process, key, map),
                        &arc_process,
                        map,
                        format!("map ({}) is not a map", map)
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}
