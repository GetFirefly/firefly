use super::*;

use proptest::prop_assert_eq;
use proptest::test_runner::{Config, TestRunner};

use crate::otp::erlang;
use crate::process;
use crate::scheduler::with_process_arc;
use crate::test::{registered_name, strategy};

mod unregister_1;
mod whereis_1;
mod xor_2;
