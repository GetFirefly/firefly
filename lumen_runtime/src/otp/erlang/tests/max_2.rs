use super::FirstSecond::{First, Second};
use super::*;

use proptest::strategy::Strategy;

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

#[test]
fn max_is_first_if_first_is_greater_than_or_equal_to_second() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term(arc_process.clone()),
                    strategy::term(arc_process.clone()),
                )
                    .prop_filter("First must be >= second", |(first, second)| second <= first),
                |(first, second)| {
                    prop_assert_eq!(erlang::max_2(first, second), first);

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn max_is_second_if_first_is_less_than_second() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term(arc_process.clone()),
                    strategy::term(arc_process.clone()),
                )
                    .prop_filter("First must be <= second", |(first, second)| first < second),
                |(first, second)| {
                    prop_assert_eq!(erlang::max_2(first, second), second);

                    Ok(())
                },
            )
            .unwrap();
    });
}

fn max<F, S>(first: F, second: S, which: FirstSecond)
where
    F: FnOnce(&ProcessControlBlock) -> Term,
    S: FnOnce(Term, &ProcessControlBlock) -> Term,
{
    with_process(|process| {
        let first = first(&process);
        let second = second(first, &process);

        let max = erlang::max_2(first, second);

        let expected = match which {
            First => first,
            Second => second,
        };

        // expected value
        assert_eq!(max, expected);
    });
}
