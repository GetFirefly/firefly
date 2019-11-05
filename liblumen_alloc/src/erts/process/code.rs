pub mod stack;

use alloc::sync::Arc;
use core::fmt::{self, Debug};

use crate::erts::exception::{system, Exception};
use crate::erts::process::Process;

pub type Result = core::result::Result<(), system::Exception>;
pub type Code = fn(&Arc<Process>) -> Result;

#[derive(Clone, Copy)]
pub struct DebuggableCode(pub Code);

impl Debug for DebuggableCode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:x}", self.0 as usize)
    }
}

pub fn result_from_exception(process: &Process, exception: Exception) -> Result {
    match exception {
        Exception::Runtime(runtime_exception) => {
            process.exception(runtime_exception);

            Ok(())
        }
        Exception::System(system_exception) => Err(system_exception),
    }
}
