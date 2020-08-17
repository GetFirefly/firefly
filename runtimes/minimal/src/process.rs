use liblumen_alloc::erts::exception::AllocResult;
use liblumen_alloc::erts::process::{FrameWithArguments, Process};

pub use lumen_rt_core::process::{current_process, monitor, spawn};

#[export_name = "lumen_rt_process_runnable"]
pub fn runnable<'a>(
    process: &Process,
    _frames_with_arguments_fn: Box<dyn Fn(&Process) -> AllocResult<Vec<FrameWithArguments>> + 'a>,
) -> AllocResult<()> {
    process.runnable(move |_process| Ok(()))
}
