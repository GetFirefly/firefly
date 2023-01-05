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
use proptest::strategy::{Just, Strategy};
use proptest::test_runner::{Config, TestRunner};

use firefly_rt::term::{Integer, Term};

use crate::erlang::are_exactly_not_equal_2::result;
use crate::test::strategy;
use crate::test::with_process_arc;
