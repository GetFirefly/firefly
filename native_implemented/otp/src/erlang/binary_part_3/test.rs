mod with_bitstring;

use std::convert::TryInto;
use std::sync::Arc;

use proptest::strategy::Just;
use proptest::test_runner::TestCaseResult;
use proptest::{prop_assert, prop_assert_eq};

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::erlang::binary_part_3::result;

// `without_bitstring_errors_badarg` in integration tests
