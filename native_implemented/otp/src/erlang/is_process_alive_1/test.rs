mod with_pid;

use proptest::test_runner::{Config, TestRunner};
use proptest::{prop_assert, prop_assert_eq};

use crate::erlang::is_process_alive_1::result;
use crate::test::strategy;
use crate::test::with_process_arc;

// `without_pid_errors_badarg` in integration tests
