//! Contains infrastructure for configuring the compiler, including parsing
//! command-line options.
mod debug;
mod input;
mod optimization;
mod options;
mod output;
mod project;
mod sanitizer;

pub use self::debug::DebugInfo;
pub use self::input::Input;
pub use self::optimization::{LinkerPluginLto, Lto, LtoCli, OptLevel, Passes};
pub use self::options::{
    CodegenOptions, DebuggingOptions, OptionGroup, OptionInfo, Options, ParseOption,
    ShowOptionGroupHelp,
};
pub use self::output::{calculate_outputs, Emit, OutputType, OutputTypeError, OutputTypes};
pub use self::project::ProjectType;
pub use self::sanitizer::Sanitizer;
