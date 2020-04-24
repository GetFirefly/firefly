mod with_atom_first;
mod with_big_integer_first;
mod with_empty_list_first;
mod with_external_pid_first;
mod with_float_first;
mod with_heap_binary_first;
mod with_list_first;
mod with_local_pid_first;
mod with_local_reference_first;
mod with_map_first;
mod with_small_integer_first;
mod with_subbinary_first;
mod with_tuple_first;

use proptest::prop_assert_eq;
use proptest::strategy::Strategy;
use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::erlang::min_2::result;
use crate::test::with_process;
use crate::test::FirstSecond::*;
use crate::test::{external_arc_node, strategy, FirstSecond};

#[test]
fn min_is_first_if_first_is_less_than_or_equal_to_second() {
    run!(
        |arc_process| {
            (
                strategy::term(arc_process.clone()),
                strategy::term(arc_process.clone()),
            )
                .prop_filter("First must be <= second", |(first, second)| first <= second)
        },
        |(first, second)| {
            prop_assert_eq!(result(first, second), first);

            Ok(())
        },
    );
}

#[test]
fn min_is_second_if_first_is_greater_than_second() {
    TestRunner::new(Config::with_source_file(file!()))
        .run(
            &strategy::process()
                .prop_flat_map(|arc_process| {
                    (
                        strategy::term(arc_process.clone()),
                        strategy::term(arc_process.clone()),
                    )
                })
                .prop_filter("First must be > second", |(first, second)| second < first),
            |(first, second)| {
                prop_assert_eq!(result(first, second), second);

                Ok(())
            },
        )
        .unwrap();
}

fn min<F, S>(first: F, second: S, which: FirstSecond)
where
    F: FnOnce(&Process) -> Term,
    S: FnOnce(Term, &Process) -> Term,
{
    with_process(|process| {
        let first = first(&process);
        let second = second(first, &process);

        let min = result(first, second);

        let expected = match which {
            First => first,
            Second => second,
        };

        // expected value
        assert_eq!(min, expected);
    });
}
