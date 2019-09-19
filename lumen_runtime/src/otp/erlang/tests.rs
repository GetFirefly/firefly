use super::*;

use proptest::prop_assert_eq;
use proptest::test_runner::{Config, TestRunner};

use crate::otp::erlang;
use crate::scheduler::with_process_arc;
use crate::test::strategy;

mod xor_2;
