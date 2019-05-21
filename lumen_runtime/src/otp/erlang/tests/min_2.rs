use super::FirstSecond::{First, Second};
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
    F: FnOnce(&Process) -> Term,
    S: FnOnce(Term, &Process) -> Term,
{
    with_process(|process| {
        let first = first(&process);
        let second = second(first, &process);

        let min = erlang::min_2(first, second);

        let expected = match which {
            First => first,
            Second => second,
        };

        // expected value
        assert_eq!(min, expected);
        // expected which when equal.  Can only really detect flipping for boxed with different
        // addresses
        assert_eq!(min.tagged, expected.tagged);
    });
}
