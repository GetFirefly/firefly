use core::convert::{From, Into};
use core::fmt;

use inkwell::targets::FileType;

pub use inkwell::targets::ByteOrdering;
pub use inkwell::targets::CodeModel;
pub use inkwell::targets::RelocMode;

pub use inkwell::AddressSpace;
pub use inkwell::AtomicOrdering;
pub use inkwell::FloatPredicate;
pub use inkwell::IntPredicate;
pub use inkwell::GlobalVisibility;
pub use inkwell::OptimizationLevel;
pub use inkwell::ThreadLocalMode;

/// An extension of `inkwell::targets::FileType` which adds the IR type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputType {
    IR,
    Assembly,
    Object,
}
impl fmt::Display for OutputType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            OutputType::IR => f.write_str("ll"),
            OutputType::Assembly => f.write_str("s"),
            OutputType::Object => f.write_str("o"),
        }
    }
}
impl From<FileType> for OutputType {
    fn from(ty: FileType) -> Self {
        match ty {
            FileType::Assembly => OutputType::Assembly,
            FileType::Object => OutputType::Object,
        }
    }
}
impl Into<FileType> for OutputType {
    fn into(self) -> FileType {
        match self {
            OutputType::Assembly => FileType::Assembly,
            OutputType::Object => FileType::Object,
            OutputType::IR => panic!("inkwell::targets::FileType does not support the IR type"),
        }
    }
}
