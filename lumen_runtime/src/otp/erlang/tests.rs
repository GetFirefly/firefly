use super::*;

use proptest::prop_assert_eq;
use proptest::strategy::{Just, Strategy};
use proptest::test_runner::{Config, TestRunner};

use crate::otp::erlang;
use crate::process;
use crate::scheduler::with_process_arc;
use crate::test::{registered_name, strategy};

mod throw_1;
mod tl_1;
mod tuple_size_1;
mod tuple_to_list_1;
mod unregister_1;
mod whereis_1;
mod xor_2;
