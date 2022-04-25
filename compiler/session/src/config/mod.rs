//! Contains infrastructure for configuring the compiler, including parsing
//! command-line options.
mod app;
mod cfguard;
mod debug;
mod input;
mod linker;
mod mlir;
mod optimization;
mod options;
mod output;
mod project;
mod sanitizer;

pub use self::app::*;
pub use self::cfguard::*;
pub use self::debug::*;
pub use self::input::{Input, InputType};
pub use self::linker::*;
pub use self::mlir::*;
pub use self::optimization::*;
pub use self::options::{
    CodegenOptions, DebuggingOptions, OptionGroup, OptionInfo, Options, ParseOption,
    ShowOptionGroupHelp,
};
pub use self::output::{calculate_outputs, OutputType, OutputTypeError, OutputTypes};
pub use self::project::*;
pub use self::sanitizer::*;
