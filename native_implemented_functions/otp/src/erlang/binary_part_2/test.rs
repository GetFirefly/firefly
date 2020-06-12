mod with_bitstring;

use std::convert::TryInto;
use std::sync::Arc;

use proptest::strategy::Just;
use proptest::test_runner::TestCaseResult;
use proptest::{prop_assert, prop_assert_eq};

use liblumen_alloc::erts::process::alloc::TermAlloc;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::erlang::binary_part_2::result;
use crate::test::{strategy, total_byte_len};

#[test]
fn without_bitstring_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_bitstring(arc_process.clone()),
            )
        },
        |(arc_process, binary)| {
            let start_length = {
                arc_process
                    .tuple_from_slice(&[
                        arc_process.integer(0).unwrap(),
                        arc_process.integer(0).unwrap(),
                    ])
                    .unwrap()
            };

            prop_assert_badarg!(
                result(&arc_process, binary, start_length),
                format!(
                    "binary ({}) must be a binary or bitstring with at least 1 full byte",
                    binary
                )
            );

            Ok(())
        },
    );
}
