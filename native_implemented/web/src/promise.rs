use std::sync::Arc;

use js_sys::Promise;

use liblumen_core::locks::Mutex;

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

pub fn to_term(promise: Promise, process: &Process) -> Term {
    process.resource(Arc::new(Mutex::new(promise)))
}
