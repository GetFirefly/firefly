mod with_bitstring;

use std::convert::TryInto;

use proptest::strategy::Just;
use proptest::test_runner::{Config, TestRunner};
use proptest::{prop_assert, prop_assert_eq};

use liblumen_alloc::badarg;
use liblumen_alloc::erts::process::alloc::TermAlloc;
use liblumen_alloc::erts::term::prelude::*;

use crate::otp::erlang::binary_part_3::native;
use crate::scheduler::with_process_arc;
use crate::test::{strategy, total_byte_len};

#[test]
fn without_bitstring_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_not_bitstring(arc_process.clone()),
                |binary| {
                    let start = arc_process.integer(0).unwrap();
                    let length = arc_process.integer(0).unwrap();

                    prop_assert_eq!(
                        native(&arc_process, binary, start, length),
                        Err(badarg!(&arc_process).into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}
