pub mod spawn;

use liblumen_alloc::erts::process::{Process, Status};

pub use lumen_rt_core::process::{current_process, monitor};

use crate::scheduler::{self, Scheduler};

pub fn runnable(process: &Process) {
    let mut writable_status = process.status.write();

    if *writable_status == Status::Unrunnable {
        unsafe {
            let mut stack = process.stack.lock();
            let mut registers = process.registers.lock();
            stack.push64(return_continuation as u64);

            registers.rsp = stack.top as u64;
            registers.rbp = stack.top as u64;
        }

        *writable_status = Status::Runnable
    }
}

#[naked]
#[inline(never)]
#[cfg(all(unix, target_arch = "x86_64"))]
pub unsafe extern "C" fn return_continuation() {
    let f: fn() -> () = return_function;
    asm!("
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
