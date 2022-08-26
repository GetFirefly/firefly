pub use firefly_llvm::support::*;

/// Corresponds to `mlir::LogicalResult`, used to indicate
/// success/failure of an operation
#[repr(u8)]
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum LogicalResult {
    Failure = 0,
    Success = 1,
}
impl Into<bool> for LogicalResult {
    #[inline(always)]
    fn into(self) -> bool {
        self == LogicalResult::Success
    }
}
