mod with_arity_2;

use super::*;

use proptest::prop_oneof;

#[test]
fn without_arity_2_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                (1_usize..=3_usize),
                strategy::term(arc_process.clone()),
                (Just(arc_process.clone()), prop_oneof![Just(1), Just(3)]).prop_flat_map(
                    |(arc_process, len)| {
                        strategy::term::tuple::intermediate(
                            strategy::term(arc_process.clone()),
                            (len..=len).into(),
                            arc_process.clone(),
                        )
                    },
                ),
            )
        },
        |(arc_process, arity_usize, default_value, element)| {
            let arity = arc_process.integer(arity_usize).unwrap();
            let init_list = arc_process.list_from_slice(&[element]).unwrap();

            prop_assert_badarg!(
                result(&arc_process, arity, default_value, init_list),
                format!(
                    "init list ({}) element ({}) is a tuple, but not 2-arity",
                    init_list, element
                )
            );

            Ok(())
        },
    );
}
