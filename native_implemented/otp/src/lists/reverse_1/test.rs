mod with_proper_list;

use proptest::prop_assert_eq;
use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::erts::term::prelude::Term;

use crate::lists::reverse_1::result;
use crate::test::strategy;
use crate::test::with_process_arc;

#[test]
fn without_proper_list_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_not_proper_list(arc_process.clone()),
                |list| {
                    prop_assert_badarg!(
                        result(&arc_process, list),
                        format!("list ({}) is not a proper list", list)
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}
