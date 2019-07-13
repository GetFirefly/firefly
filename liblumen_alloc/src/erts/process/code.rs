pub mod stack;

use alloc::sync::Arc;

use crate::erts::exception::system;
use crate::erts::process::ProcessControlBlock;

pub type Result = core::result::Result<(), system::Exception>;
pub type Code = fn(&Arc<ProcessControlBlock>) -> Result;
