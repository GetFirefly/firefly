mod with_safe;

use proptest::strategy::Just;

use firefly_rt::process::Process;
use firefly_rt::term::Term;

use crate::erlang::binary_to_term_2::result;
use crate::test::strategy;

// `with_used_with_binary_returns_how_many_bytes_were_consumed_along_with_term` in integration tests
