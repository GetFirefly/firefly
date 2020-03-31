pub mod anonymous;
pub mod export;

use std::sync::Arc;

use liblumen_alloc::erts::process::{code, Process};

/// A stub that just puts the init process into `Status::Waiting`, so it remains alive without
/// wasting CPU cycles
pub fn wait(arc_process: &Arc<Process>) -> code::Result {
    Arc::clone(arc_process).wait();

    Ok(())
}
