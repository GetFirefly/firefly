use llvm_sys::prelude::LLVMBool;
use llvm_sys::target_machine::{LLVMCodeGenFileType, LLVMCodeGenOptLevel};

/// Represents the type of output to generate
pub enum OutputType {
    IR,
    Assembly,
    Object,
}
impl std::fmt::Display for OutputType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
            OutputType::IR => f.write_str("ll"),
            OutputType::Assembly => f.write_str("S"),
            OutputType::Object => f.write_str("o"),
        }
    }
}
impl std::convert::From<LLVMCodeGenFileType> for OutputType {
    fn from(ty: LLVMCodeGenFileType) -> Self {
        match ty {
            LLVMCodeGenFileType::LLVMAssemblyFile => OutputType::Assembly,
            LLVMCodeGenFileType::LLVMObjectFile => OutputType::Object,
        }
    }
}
impl std::convert::Into<LLVMCodeGenFileType> for OutputType {
    fn into(self) -> LLVMCodeGenFileType {
        match self {
            OutputType::Assembly => LLVMCodeGenFileType::LLVMAssemblyFile,
            OutputType::Object => LLVMCodeGenFileType::LLVMObjectFile,
            OutputType::IR => panic!("LLVMCodeGenFileType does not support the IR type"),
        }
    }
}

/// Represents the amount of optimization to apply during codegen
pub enum Optimization {
    None,
    Less,
    Default,
    Aggressive,
}
impl std::convert::From<LLVMCodeGenOptLevel> for Optimization {
    fn from(level: LLVMCodeGenOptLevel) -> Self {
        match level {
            LLVMCodeGenOptLevel::LLVMCodeGenLevelNone => Optimization::None,
            LLVMCodeGenOptLevel::LLVMCodeGenLevelLess => Optimization::Less,
            LLVMCodeGenOptLevel::LLVMCodeGenLevelDefault => Optimization::Default,
            LLVMCodeGenOptLevel::LLVMCodeGenLevelAggressive => Optimization::Aggressive,
        }
    }
}
impl std::convert::Into<u32> for Optimization {
    fn into(self) -> u32 {
        match self {
            Optimization::None => 0,
            Optimization::Less => 1,
            Optimization::Default => 2,
            Optimization::Aggressive => 3,
        }
    }
}
impl std::convert::Into<LLVMCodeGenOptLevel> for Optimization {
    fn into(self) -> LLVMCodeGenOptLevel {
        match self {
            Optimization::None => LLVMCodeGenOptLevel::LLVMCodeGenLevelNone,
            Optimization::Less => LLVMCodeGenOptLevel::LLVMCodeGenLevelLess,
            Optimization::Default => LLVMCodeGenOptLevel::LLVMCodeGenLevelDefault,
            Optimization::Aggressive => LLVMCodeGenOptLevel::LLVMCodeGenLevelAggressive,
        }
    }
}

pub enum Bool {
    True,
    False,
}
impl std::convert::From<LLVMBool> for Bool {
    fn from(b: LLVMBool) -> Bool {
        if (b as i32) == 0 {
            Bool::True
        } else {
            Bool::False
        }
    }
}
impl std::convert::Into<LLVMBool> for Bool {
    fn into(self) -> LLVMBool {
        match self {
            Bool::True => (0 as LLVMBool),
            Bool::False => (1 as LLVMBool),
        }
    }
}
