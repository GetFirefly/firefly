use std::sync::Arc;

use liblumen_alloc::erts::process::{code, Process};

use locate_code::locate_code;

/// A stub that just puts the init process into `Status::Waiting`, so it remains alive without
/// wasting CPU cycles
#[locate_code]
pub fn code(arc_process: &Arc<Process>) -> code::Result {
    Arc::clone(arc_process).wait();

    Ok(())
}
