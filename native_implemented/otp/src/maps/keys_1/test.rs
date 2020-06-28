mod with_map;

use proptest::prop_assert_eq;
use proptest::strategy::Strategy;
use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::atom;
use liblumen_alloc::erts::term::prelude::*;

use crate::maps::keys_1::result;
use crate::test::strategy;
use crate::test::with_process_arc;

#[test]
fn without_map_errors_badmap() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&(strategy::term::is_not_map(arc_process.clone())), |map| {
                prop_assert_badmap!(result(&arc_process, map), &arc_process, map);

                Ok(())
            })
            .unwrap();
    });
}
