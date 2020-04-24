mod with_improper_list;
mod with_proper_non_empty_list;

use super::*;

#[test]
fn with_empty_tuple_list_returns_false() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term(arc_process.clone()),
                    strategy::term::index::is_one_based(arc_process.clone()),
                ),
                |(key, one_based_index)| {
                    let tuple_list = Term::NIL;

                    prop_assert_eq!(result(key, one_based_index, tuple_list), Ok(false.into()));

                    Ok(())
                },
            )
            .unwrap();
    });
}
