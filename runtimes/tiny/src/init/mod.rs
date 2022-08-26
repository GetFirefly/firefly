use firefly_rt::function::ErlangResult;
use firefly_rt::term::{ListBuilder, OpaqueTerm};

use crate::env;
use crate::scheduler;

extern "C-unwind" {
    #[allow(improper_ctypes)]
    #[allow(improper_ctypes_definitions)]
    #[link_name = "init:boot/1"]
    fn boot(argv: OpaqueTerm) -> ErlangResult;
}

/// This function acts as the entry point for the top-level `init` process.
///
/// Its job is to preprocess command-line arguments and boot the system.
/// The actual boot process is handled in `init:boot/1`, or if substituted with
/// a different module, `Module:boot/1`.
///
/// NOTE: When this function is invoked, it is on the stack of the new process, not the scheduler.
#[allow(improper_ctypes_definitions)]
pub(crate) extern "C-unwind" fn start() -> ErlangResult {
    scheduler::with_current_process(|process| {
        let argv = env::argv();
        let args = {
            let mut builder = ListBuilder::new(process);
            for arg in argv.iter().rev().copied() {
                builder.push(arg.into()).unwrap();
            }
            builder
                .finish()
                .map(|ptr| ptr.into())
                .unwrap_or(OpaqueTerm::NIL)
        };
        unsafe { boot(args) }
    })
}
