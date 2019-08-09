pub mod stack;

use alloc::sync::Arc;

use crate::erts::exception::{system, Exception};
use crate::erts::process::ProcessControlBlock;

pub type Result = core::result::Result<(), system::Exception>;
pub type Code = fn(&Arc<ProcessControlBlock>) -> Result;

pub fn result_from_exception(process: &ProcessControlBlock, exception: Exception) -> Result {
    match exception {
        Exception::Runtime(runtime_exception) => {
            process.exception(runtime_exception);

            Ok(())
        }
        Exception::System(system_exception) => Err(system_exception),
    }
}
