pub mod stack;

use alloc::sync::Arc;

use crate::erts::exception::{Exception, SystemException};
use crate::erts::process::Process;

pub type Result = core::result::Result<(), SystemException>;
pub type Code = fn(&Arc<Process>) -> Result;

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
