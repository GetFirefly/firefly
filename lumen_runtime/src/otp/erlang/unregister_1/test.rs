mod with_atom_name;

use std::convert::TryInto;

use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::erts::term::prelude::*;

use crate::otp::erlang;
use crate::otp::erlang::unregister_1::native;
use crate::scheduler::with_process_arc;
use crate::test::{registered_name, strategy};
use crate::{process, registry};

#[test]
fn without_atom_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::is_not_atom(arc_process.clone()), |name| {
                prop_assert_badarg!(native(name), format!("name ({}) must be an atom", name));

                Ok(())
            })
            .unwrap();
    });
}
