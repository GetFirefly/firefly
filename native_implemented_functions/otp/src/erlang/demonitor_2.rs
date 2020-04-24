mod options;

// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::runtime::process::monitor::is_down;
use crate::runtime::registry::pid_to_process;

use native_implemented_function::native_implemented_function;

use crate::erlang::demonitor_2::options::Options;

#[native_implemented_function(demonitor/2)]
pub fn result(process: &Process, reference: Term, options: Term) -> exception::Result<Term> {
    let reference_reference = term_try_into_local_reference!(reference)?;
    let options_options: Options = options.try_into()?;

    demonitor(process, &reference_reference, options_options)
}

// Private

pub(in crate::erlang) fn demonitor(
    monitoring_process: &Process,
    reference: &Reference,
    Options { flush, info }: Options,
) -> exception::Result<Term> {
    match monitoring_process.demonitor(reference) {
        Some(monitored_pid) => {
            match pid_to_process(&monitored_pid) {
                Some(monitored_arc_proces) => match monitored_arc_proces.demonitored(reference) {
                    Some(monitoring_pid) => assert_eq!(monitoring_process.pid(), monitoring_pid),
                    None => (),
                },
                None => (),
            }

            if flush {
                let flushed = self::flush(monitoring_process, reference);

                if info && flushed {
                    Ok(false.into())
                } else {
                    Ok(true.into())
                }
            } else {
                Ok(true.into())
            }
        }
        None => {
            if info {
                Ok(false.into())
            } else {
                Ok(true.into())
            }
        }
    }
}

fn flush(monitoring_process: &Process, reference: &Reference) -> bool {
    monitoring_process
        .mailbox
        .lock()
        .borrow_mut()
        .flush(|message| is_down(message, reference), monitoring_process)
}
