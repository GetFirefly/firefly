pub mod stack;

use alloc::sync::Arc;
use core::fmt::{self, Debug};

use crate::erts::exception::{Exception, SystemException};
use crate::erts::process::Process;

pub type Result = core::result::Result<(), SystemException>;
pub type Code = fn(&Arc<Process>) -> Result;

#[derive(Clone, Copy)]
pub struct DebuggableCode(pub Code);

impl Debug for DebuggableCode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:x}", self.0 as usize)
    }
}

pub fn result_from_exception<P>(process: P, exception: Exception) -> Result
where
    P: AsRef<Process>,
{
    match exception {
        Exception::Runtime(err) => {
            process.as_ref().exception(err);

            Ok(())
        }
        Exception::System(err) => Err(err),
    }
}
