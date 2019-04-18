use super::FirstSecond::*;
use super::*;

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

fn min<F, S>(first: F, second: S, which: FirstSecond)
where
    F: FnOnce(&mut Process) -> Term,
    S: FnOnce(Term, &mut Process) -> Term,
{
    with_process(|mut process| {
        let first = first(&mut process);
        let second = second(first, &mut process);

        assert_eq!(
            erlang::min_2(first, second),
            match which {
                First => first,
                Second => second,
            }
        );
    });
}
