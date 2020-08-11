mod with_safe;

use proptest::strategy::Just;

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::{Atom, Term};

use crate::erlang::binary_to_term_2::result;
use crate::test::strategy;

// `with_used_with_binary_returns_how_many_bytes_were_consumed_along_with_term` in integration tests
