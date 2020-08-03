mod with_one_based_index;

use proptest::prop_assert_eq;
use proptest::strategy::{Just, Strategy};
use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::erts::term::prelude::*;

use crate::lists::keymember_3::result;
use crate::test::strategy;
use crate::test::with_process_arc;

#[test]
fn without_one_based_index_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term(arc_process.clone()),
                    strategy::term::index::is_not_one_based(arc_process.clone()),
                    strategy::term::list::proper(arc_process.clone()),
                ),
                |(key, one_based_index, tuple_list)| {
                    prop_assert_badarg!(
                        result(key, one_based_index, tuple_list),
                        format!("index ({}) is not a 1-based integer", one_based_index)
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}
