#![deny(warnings)]

pub mod code;
mod exec;
mod module;
pub use module::NativeModule;
pub mod call_result;
mod native;
pub use lumen_rt_core as runtime;
mod vm;

#[cfg(test)]
mod tests;

use self::vm::VMState;
use lazy_static::lazy_static;

lazy_static! {
    pub static ref VM: VMState = VMState::new();
}
