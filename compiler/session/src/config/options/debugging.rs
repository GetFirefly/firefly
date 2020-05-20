use liblumen_target::{MergeFunctions, RelroLevel};

use liblumen_compiler_macros::option_group;

use crate::config::*;

#[option_group(name = "debugging", short = "Z", help = "Available debugging flags")]
#[derive(Debug, Clone, Default)]
pub struct DebuggingOptions {
    #[option]
    /// Measure time of each lumen pass
    pub time_passes: bool,
    #[option(hidden(true))]
    /// Measure time of each LLVM pass
    pub time_mlir_passes: bool,
    #[option(hidden(true))]
    /// Measure time of each LLVM pass
    pub time_llvm_passes: bool,
    #[option(hidden(true))]
    /// Gather statistics about the input
    pub input_stats: bool,
    #[option]
    /// Generate comments into the assembly (may change behavior)
    pub asm_comments: bool,
    #[option]
    /// Verify LLVM IR
    pub verify_llvm_ir: bool,
    #[option(hidden(true))]
    /// Gather metadata statistics
    pub meta_stats: bool,
    #[option(hidden(true))]
    /// Print the arguments passed to the linker
    pub print_link_args: bool,
    #[option(hidden(true))]
    /// Prints the LLVM optimization passes being run
    pub print_mlir_passes: bool,
    #[option(hidden(true))]
    /// Prints the LLVM optimization passes being run
    pub print_llvm_passes: bool,
    #[option(default_value("1"), takes_value(true), value_name("N"))]
    /// Use a thread pool with N threads
    pub threads: u64,
    #[option]
    /// Parse only; do not compile, assemble, or link
    pub parse_only: bool,
    #[option]
    /// Run all passes except codegen; no output
    pub no_codegen: bool,
    #[option(hidden(true))]
    /// Print some performance-related statistics
    pub perf_stats: bool,
    #[option]
    /// Pass `-install_name @rpath/...` to the macOS linker
    pub osx_rpath_install_name: bool,
    #[option(
        takes_value(true),
        possible_values("address", "leak", "memory", "thread")
    )]
    /// Use a sanitizer
    pub sanitizer: Option<Sanitizer>,
    #[option(takes_value(true), value_name("ARG"))]
    /// A single extra argument to prepend the linker invocation
    /// can be used more than once
    pub pre_link_arg: Vec<String>,
    #[option(takes_value(true), value_name("ARGS"), require_delimiter(true))]
    /// Extra arguments to prepend to the linker invocation (space separated)
    pub pre_link_args: Option<Vec<String>>,
    #[option(takes_value(true), possible_values("full", "partial", "off", "none"))]
    /// Choose which RELRO level to use")
    pub relro_level: Option<RelroLevel>,
    #[option(hidden(true))]
    /// Generate a graphical HTML report of time spent in codegen and LLVM
    pub codegen_time_graph: bool,
    #[option(hidden(true))]
    /// Enable ThinLTO when possible
    pub thinlto: Option<bool>,
    #[option(hidden(true))]
    /// Embed LLVM bitcode in object files
    pub embed_bitcode: bool,
    #[option]
    /// Tell the linker to strip debuginfo when building without debuginfo enabled
    pub strip_debuginfo_if_disabled: Option<bool>,
    #[option(hidden(true))]
    /// Don't run LLVM in parallel (while keeping codegen-units and ThinLTO)
    pub no_parallel_llvm: bool,
    #[option(
        takes_value(true),
        possible_values("disabled", "trampolines", "aliases"),
        hidden(true)
    )]
    /// Control the operation of the MergeFunctions LLVM pass, taking
    /// the same values as the target option of the same name
    pub merge_functions: Option<MergeFunctions>,
    #[option(hidden(true))]
    /// Emit a section containing stack size metadata
    pub emit_stack_sizes: bool,
}
