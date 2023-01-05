mod with_bitstring;

use std::sync::Arc;

use proptest::strategy::Just;
use proptest::test_runner::TestCaseResult;
use proptest::{prop_assert, prop_assert_eq};

use firefly_rt::process::Process;
use firefly_rt::term::Term;

use crate::erlang::binary_part_3::result;

// `without_bitstring_errors_badarg` in integration tests
