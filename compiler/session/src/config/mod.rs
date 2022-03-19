//! Contains infrastructure for configuring the compiler, including parsing
//! command-line options.
mod app;
mod cfguard;
mod debug;
mod input;
mod mlir;
mod optimization;
mod options;
mod output;
mod project;
mod sanitizer;

pub use self::app::App;
pub use self::cfguard::CFGuard;
pub use self::debug::{DebugInfo, Strip};
pub use self::input::{Input, InputType};
pub use self::mlir::MlirDebugPrinting;
pub use self::optimization::{LinkerPluginLto, Lto, LtoCli, OptLevel, Passes};
pub use self::options::{
    CodegenOptions, DebuggingOptions, OptionGroup, OptionInfo, Options, ParseOption,
    ShowOptionGroupHelp,
};
pub use self::output::{calculate_outputs, OutputType, OutputTypeError, OutputTypes};
pub use self::project::ProjectType;
pub use self::sanitizer::Sanitizer;
