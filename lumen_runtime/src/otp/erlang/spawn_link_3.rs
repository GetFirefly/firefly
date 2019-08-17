// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::sync::Arc;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::exception::system::Alloc;
use liblumen_alloc::erts::process::code::stack::frame::Placement;
use liblumen_alloc::erts::process::code::{self, result_from_exception};
use liblumen_alloc::erts::process::ProcessControlBlock;
use liblumen_alloc::erts::term::Term;

use crate::otp::erlang::spawn_linkage_3;
use crate::process::Linkage;

pub fn native(
    process_control_block: &ProcessControlBlock,
    module: Term,
    function: Term,
    arguments: Term,
) -> exception::Result {
    spawn_linkage_3::native(
        process_control_block,
        Linkage::Link,
        module,
        function,
        arguments,
    )
}

pub fn place_frame_with_arguments(
    process: &ProcessControlBlock,
    placement: Placement,
    module: Term,
    function: Term,
    arguments: Term,
) -> Result<(), Alloc> {
    spawn_linkage_3::place_frame_with_arguments(
        process,
        placement,
        Linkage::Link,
        module,
        function,
        arguments,
    )
}

// Private

pub(in crate::otp::erlang) fn code(arc_process: &Arc<ProcessControlBlock>) -> code::Result {
    arc_process.reduce();

    let module = arc_process.stack_pop().unwrap();
    let function = arc_process.stack_pop().unwrap();
    let arguments = arc_process.stack_pop().unwrap();

    match native(arc_process, module, function, arguments) {
        Ok(child_pid) => {
            arc_process.return_from_call(child_pid)?;

            ProcessControlBlock::call_code(arc_process)
        }
        Err(exception) => result_from_exception(arc_process, exception),
    }
}
