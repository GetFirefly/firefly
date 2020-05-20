pub mod spawn;

use liblumen_alloc::erts::exception::AllocResult;
use liblumen_alloc::erts::process::{FrameWithArguments, Process};

pub use lumen_rt_core::process::{current_process, monitor};

use crate::scheduler::{self, Scheduler};

pub fn runnable<F>(process: &Process, frames_with_arguments_fn: F) -> AllocResult<()>
where
    F: FnOnce(&Process) -> AllocResult<Vec<FrameWithArguments>>,
{
    process.runnable(|process| {
        unsafe {
            let mut stack = process.stack.lock();
            let mut registers = process.registers.lock();

            let frames_with_arguments = frames_with_arguments_fn(process)?;

            for FrameWithArguments { frame, uses_returned, arguments  } in frames_with_arguments.into_iter().rev() {
                if uses_returned {
                    unimplemented!("Cannot generate code to use returned value dynamically");
                }

                if arguments.is_empty() {
                    stack.push_frame(&frame);
                } else {
                    unimplemented!("Cannot generate code to use heap stack values ({:?}) when calling functions", arguments);
                }
            }

            stack.push64(return_continuation as u64);

            registers.rsp = stack.top as u64;
            registers.rbp = stack.top as u64;
        }

        Ok(())
    })
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
