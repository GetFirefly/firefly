pub mod code;
mod exec;
mod module;
pub use module::NativeModule;
mod native;
mod vm;
pub mod call_result;

#[cfg(test)]
mod tests;

use self::vm::VMState;
use lazy_static::lazy_static;

lazy_static! {
    pub static ref VM: VMState = VMState::new();
}
