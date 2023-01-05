mod with_atom_left;
mod with_big_integer_left;
mod with_empty_list_left;
mod with_external_pid_left;
mod with_float_left;
mod with_function_left;
mod with_heap_binary_left;
mod with_list_left;
mod with_local_pid_left;
mod with_local_reference_left;
mod with_map_left;
mod with_small_integer_left;
mod with_subbinary_left;
mod with_tuple_left;

use std::ptr::NonNull;

use proptest::arbitrary::any;
use proptest::prop_assert_eq;
use proptest::strategy::Just;
use proptest::test_runner::{Config, TestRunner};

use firefly_rt::term::{Integer, Term};

use crate::erlang::are_equal_after_conversion_2::result;
use crate::test::strategy;
use crate::test::with_process_arc;

#[test]
fn without_numbers_are_not_equal_after_conversion_if_not_equal_before_conversion() {
    run!(
        |arc_process| {
            (
                strategy::term::is_not_number(arc_process.clone()),
                strategy::term::is_not_number(arc_process),
            )
                .prop_filter(
                    "Left must not equal right before conversion",
                    |(left, right)| left != right,
                )
        },
        |(left, right)| {
            prop_assert_eq!(result(left, right), false.into());

            Ok(())
        },
    );
}
