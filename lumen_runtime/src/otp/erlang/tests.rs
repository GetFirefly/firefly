use super::*;

use proptest::strategy::{BoxedStrategy, Just, Strategy};
use proptest::test_runner::{Config, TestRunner};
use proptest::{prop_assert, prop_assert_eq};

use liblumen_alloc::erts::term::next_pid;

use crate::otp::erlang;
use crate::process;
use crate::scheduler::{with_process, with_process_arc};
use crate::test::{has_message, registered_name, strategy, timeout_message};

mod setelement_3;
mod size_1;
mod split_binary_2;
mod start_timer_3;
mod start_timer_4;
mod subtract_list_2;
mod throw_1;
mod tl_1;
mod tuple_size_1;
mod tuple_to_list_1;
mod unregister_1;
mod whereis_1;
mod xor_2;
