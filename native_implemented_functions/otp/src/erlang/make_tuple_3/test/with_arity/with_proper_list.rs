mod with_tuples_in_init_list;

use super::*;

#[test]
fn without_tuple_in_init_list_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                (0_usize..255_usize),
                strategy::term(arc_process.clone()),
                strategy::term::is_not_tuple(arc_process.clone()),
            )
        },
        |(arc_process, arity_usize, default_value, element)| {
            let arity = arc_process.integer(arity_usize).unwrap();
            let init_list = arc_process.list_from_slice(&[element]).unwrap();

            prop_assert_badarg!(
                result(&arc_process, arity, default_value, init_list),
                format!(
                    "init list ({}) element ({}) is not {{position :: pos_integer(), term()}}",
                    init_list, element
                )
            );

            Ok(())
        },
    );
}
