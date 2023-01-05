mod with_arity;

use std::ptr::NonNull;

use proptest::strategy::Just;
use proptest::test_runner::{Config, TestRunner};
use proptest::{prop_assert, prop_assert_eq};

use firefly_rt::term::{Term, Tuple};

use crate::erlang::make_tuple_3::result;
use crate::test::strategy;

#[test]
fn without_arity_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_arity(arc_process.clone()),
                strategy::term(arc_process.clone()),
            )
        },
        |(arc_process, arity, default_value)| {
            let init_list = Term::Nil;

            prop_assert_is_not_arity!(result(&arc_process, arity, default_value, init_list), arity);

            Ok(())
        },
    );
}
