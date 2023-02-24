use std::fmt;

/// Represents what type of file will be codegen'd
///
/// Currently supports two output types:
///
/// * Textual assembly (i.e. `foo.s`)
/// * Object code (i.e. `foo.o`)
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub enum CodeGenFileType {
    Assembly = 0,
    Object,
}

/// Represents the speed optimization level to apply during codegen
///
/// The default level is equivalent to -02
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd)]
#[repr(C)]
pub enum CodeGenOptLevel {
    None,
    Less,
    Default,
    Aggressive,
}
impl Default for CodeGenOptLevel {
    fn default() -> Self {
        Self::Default
    }
}
impl fmt::Display for CodeGenOptLevel {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::None => f.write_str("none"),
            Self::Less => f.write_str("less"),
            Self::Default => f.write_str("default"),
            Self::Aggressive => f.write_str("aggressive"),
        }
    }
}

/// Represents the size optimization level to apply during codegen
///
/// The default level is equivalent to -O2
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd)]
#[repr(C)]
pub enum CodeGenOptSize {
    None,
    Less,
    Default,
    Aggressive,
}
impl Default for CodeGenOptSize {
    fn default() -> Self {
        Self::Default
    }
}
impl fmt::Display for CodeGenOptSize {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::None => f.write_str("none"),
            Self::Less => f.write_str("less"),
            Self::Default => f.write_str("default"),
            Self::Aggressive => f.write_str("aggressive"),
        }
    }
}

/// Converts the unified OptLevel enum from the frontend to the speed/size opt level enums for LLVM
pub fn to_llvm_opt_settings(cfg: firefly_session::OptLevel) -> (CodeGenOptLevel, CodeGenOptSize) {
    use firefly_session::OptLevel;
    match cfg {
        OptLevel::No => (CodeGenOptLevel::None, CodeGenOptSize::None),
        OptLevel::Less => (CodeGenOptLevel::Less, CodeGenOptSize::Less),
        OptLevel::Default => (CodeGenOptLevel::Default, CodeGenOptSize::Default),
        OptLevel::Aggressive => (CodeGenOptLevel::Aggressive, CodeGenOptSize::None),
        OptLevel::Size => (CodeGenOptLevel::Less, CodeGenOptSize::Default),
        OptLevel::SizeMin => (CodeGenOptLevel::Less, CodeGenOptSize::Aggressive),
    }
}
