pub mod monitor;

use liblumen_alloc::erts::process::Process;
use std::cell::RefCell;
use std::sync::Arc;

thread_local! {
  pub static CURRENT_PROCESS: RefCell<Option<Arc<Process>>> = RefCell::new(None);
}

pub fn current_process() -> Arc<Process> {
    CURRENT_PROCESS.with(|cp| cp.borrow().clone().expect("no process currently scheduled"))
}

pub fn maybe_current_process() -> Option<Arc<Process>> {
    CURRENT_PROCESS.with(|cp| cp.borrow().clone())
}
