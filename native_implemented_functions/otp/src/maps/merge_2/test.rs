mod with_map_map1;

use proptest::prop_assert_eq;
use proptest::strategy::{Just, Strategy};
use proptest::test_runner::{Config, TestRunner};

use crate::maps::merge_2::result;
use crate::test::strategy;

#[test]
fn without_map_map_1_errors_badmap() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_map(arc_process.clone()),
                strategy::term::is_map(arc_process.clone()),
            )
        },
        |(arc_process, map1, map2)| {
            prop_assert_badmap!(result(&arc_process, map1, map2), &arc_process, map1);

            Ok(())
        },
    );
}
