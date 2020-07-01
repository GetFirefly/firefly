mod with_positive_index;

use super::*;

#[test]
fn without_positive_index_errors_badarg_because_indexes_are_one_based() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                (1_usize..3_usize),
                strategy::term(arc_process.clone()),
                strategy::term(arc_process.clone())
                    .prop_filter("Index must not be a positive index", |index| {
                        !index.is_integer() || index <= &fixnum!(0)
                    }),
                strategy::term(arc_process.clone()),
            )
        },
        |(arc_process, arity_usize, default_value, position, element)| {
            let arity = arc_process.integer(arity_usize).unwrap();
            let init = arc_process.tuple_from_slice(&[position, element]).unwrap();
            let init_list = arc_process.list_from_slice(&[init]).unwrap();

            let r = result(&arc_process, arity, default_value, init_list);

            prop_assert_badarg!(
                r,
                format!(
                    "init list ({}) element ({}) position ({}) is not a positive integer",
                    init_list, init, position
                )
            );

            Ok(())
        },
    );
}
