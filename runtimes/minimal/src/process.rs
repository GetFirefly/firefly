use core::ptr;

use liblumen_alloc::erts::exception::AllocResult;
use liblumen_alloc::erts::process::{FrameWithArguments, Process};

pub use lumen_rt_core::process::{current_process, monitor, spawn};

use crate::scheduler::{self, Scheduler};

#[export_name = "lumen_rt_process_runnable"]
pub fn runnable<'a>(
    process: &Process,
    frames_with_arguments_fn: Box<dyn Fn(&Process) -> AllocResult<Vec<FrameWithArguments>> + 'a>,
) -> AllocResult<()> {
    process.runnable(move |process| Ok(()))
}

#[naked]
#[inline(never)]
#[cfg(all(unix, target_arch = "x86_64"))]
pub unsafe extern "C" fn return_continuation() {
    let f: fn() -> () = return_function;
    llvm_asm!("
        callq *$0
        "
    :
    : "r"(f)
    :
    : "volatile", "alignstack"
    );
}

#[inline(never)]
fn return_function() {
    scheduler::current()
        .as_any()
        .downcast_ref::<Scheduler>()
        .unwrap()
        .process_return();
}
