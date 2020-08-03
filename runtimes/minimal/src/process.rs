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
    process.runnable(move |process| {
        unsafe {
            let mut stack = process.stack.lock();

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

            // This is an obviously blatantly unsafe operation to perform, but is
            // actually safe in this case - it is never possible for the process
            // registers to be modified concurrently - the owning scheduler will
            // be the only thing modifying them, and can only do so once it has
            // taken ownership of the process to begin with, implying exclusive
            // access
            let registers = &process.registers;
            ptr::write(&registers.rsp as *const _ as *mut _, stack.top as u64);
            ptr::write(&registers.rbp as *const _ as *mut _, stack.top as u64);
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
