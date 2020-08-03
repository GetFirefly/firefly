use liblumen_target::{MergeFunctions, RelroLevel};

use liblumen_compiler_macros::option_group;

use crate::config::*;

#[option_group(name = "debugging", short = "Z", help = "Available debugging flags")]
#[derive(Debug, Clone, Default)]
pub struct DebuggingOptions {
    #[option]
    /// Generate comments into the assembly (may change behavior)
    pub asm_comments: bool,
    #[option(hidden(true))]
    /// Emit a section containing stack size metadata
    pub emit_stack_sizes: bool,
    #[option(hidden(true))]
    /// Gather statistics about the input
    pub input_stats: bool,
    #[option(default_value("true"))]
    /// Link native libraries in the linker invocation
    pub link_native_libraries: bool,
    #[option]
    /// Control whether to link Rust provided C objects/libraries or rely
    /// on C toolchain installed in the system
    pub link_self_contained: Option<bool>,
    #[option(
        takes_value(true),
        possible_values("disabled", "trampolines", "aliases"),
        hidden(true)
    )]
    /// Control the operation of the MergeFunctions LLVM pass, taking
    /// the same values as the target option of the same name
    pub merge_functions: Option<MergeFunctions>,
    #[option]
    /// Run all passes except codegen; no output
    pub no_codegen: bool,
    #[option(hidden(true))]
    /// Don't run LLVM in parallel (while keeping codegen-units and ThinLTO)
    pub no_parallel_llvm: bool,
    #[option]
    /// Pass `-install_name @rpath/...` to the macOS linker
    pub osx_rpath_install_name: bool,
    #[option]
    /// Parse only; do not compile, assemble, or link
    pub parse_only: bool,
    #[option(hidden(true))]
    /// Print some performance-related statistics
    pub perf_stats: bool,
    #[option(takes_value(true), value_name("ARG"))]
    /// A single extra argument to prepend the linker invocation
    /// can be used more than once
    pub pre_link_arg: Vec<String>,
    #[option(takes_value(true), value_name("ARGS"), require_delimiter(true))]
    /// Extra arguments to prepend to the linker invocation (space separated)
    pub pre_link_args: Option<Vec<String>>,
    #[option(hidden(true))]
    /// Print the arguments passed to the linker
    pub print_link_args: bool,
    #[option(hidden(true))]
    /// Prints the LLVM optimization passes being run
    pub print_mlir_passes: bool,
    #[option(hidden(true))]
    /// Prints the LLVM optimization passes being run
    pub print_llvm_passes: bool,
    #[option(takes_value(true), possible_values("full", "partial", "off", "none"))]
    /// Choose which RELRO level to use")
    pub relro_level: Option<RelroLevel>,
    #[option(
        takes_value(true),
        possible_values("address", "leak", "memory", "thread")
    )]
    /// Use a sanitizer
    pub sanitizer: Option<Sanitizer>,
    #[option(
        next_line_help(true),
        takes_value(true),
        value_name("TYPE"),
        default_value("none"),
        possible_values("none", "debuginfo", "symbols")
    )]
    /**
     ** Tell the linker which information to strip:
     **     none      = do not strip anything
     **     debuginfo = strip debugging information
     **     symbols   = strip debugging symbols wh
     **     _
     **/
    pub strip: Strip,
    #[option(hidden(true))]
    /// Enable ThinLTO when possible
    pub thinlto: Option<bool>,
    #[option(default_value("1"), takes_value(true), value_name("N"))]
    /// Use a thread pool with N threads
    pub threads: u64,
    #[option]
    /// Measure time of each lumen pass
    pub time_passes: bool,
    #[option(hidden(true))]
    /// Measure time of each LLVM pass
    pub time_mlir_passes: bool,
    #[option(hidden(true))]
    /// Measure time of each LLVM pass
    pub time_llvm_passes: bool,
    #[option]
    /// Verify LLVM IR
    pub verify_llvm_ir: bool,
}
