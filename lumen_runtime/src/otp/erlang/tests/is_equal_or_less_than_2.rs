use super::*;

mod with_atom_left;
mod with_big_integer_left;
mod with_empty_list_left;
mod with_external_pid_left;
mod with_float_left;
mod with_heap_binary_left;
mod with_list_left;
mod with_local_pid_left;
mod with_local_reference_left;
mod with_map_left;
mod with_small_integer_left;
mod with_subbinary_left;
mod with_tuple_left;

fn is_equal_or_less_than<L, R>(left: L, right: R, expected: bool)
where
    L: FnOnce(&mut Process) -> Term,
    R: FnOnce(Term, &mut Process) -> Term,
{
    with_process(|mut process| {
        let left = left(&mut process);
        let right = right(left, &mut process);

        assert_eq!(
            erlang::is_equal_or_less_than_2(left, right),
            expected.into()
        );
    });
}
