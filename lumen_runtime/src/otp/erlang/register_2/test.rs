mod with_atom_name;

use std::convert::TryInto;
use std::sync::Arc;

use proptest::prop_assert_eq;
use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::badarg;
use liblumen_alloc::erts::term::prelude::{Atom, Encoded, Pid};

use crate::otp::erlang;
use crate::otp::erlang::register_2::native;
use crate::scheduler::with_process_arc;
use crate::test::{registered_name, strategy};
use crate::{process, registry};

#[test]
fn without_atom_name_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_not_atom(arc_process.clone()),
                    strategy::term::pid_or_port(arc_process.clone()),
                ),
                |(name, pid_or_port)| {
                    prop_assert_eq!(
                        native(arc_process.clone(), name, pid_or_port,),
                        Err(badarg!().into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}
