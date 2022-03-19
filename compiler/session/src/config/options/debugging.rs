use std::path::PathBuf;

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
    /// Control whether to link Lumen provided C objects/libraries or rely
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
    #[option]
    /// Pass `-install_name @rpath/...` to the macOS linker
    pub osx_rpath_install_name: bool,
    #[option]
    /// Parse only; do not compile, assemble, or link
    pub parse_only: bool,
    #[option(takes_value(true), value_name("ARG"))]
    /// A single extra argument to prepend the linker invocation
    /// can be used more than once
    pub pre_link_arg: Vec<String>,
    #[option(takes_value(true), value_name("ARGS"), require_delimiter(true))]
    /// Extra arguments to prepend to the linker invocation (space separated)
    pub pre_link_args: Option<Vec<String>>,
    #[option(takes_value(true), possible_values("false", "none", "plain", "pretty"))]
    /// Enable printing of debug info when printing MLIR
    pub mlir_print_debug_info: MlirDebugPrinting,
    #[option(hidden(true))]
    /// Print the arguments passed to the linker
    pub print_link_args: bool,
    #[option(default_value("false"))]
    /// Prints MLIR IR and pass name for each optimization pass before being run
    pub print_passes_before: bool,
    #[option(default_value("false"))]
    /// Prints MLIR IR and pass name for each optimization pass after being run
    pub print_passes_after: bool,
    #[option(default_value("false"))]
    /// Only print MLIR IR and pass name after each optimization pass when the IR changes
    /// This is expected to be combined with `print_passes_after`.
    pub print_passes_on_change: bool,
    #[option(default_value("false"))]
    /// Only print MLIR IR and pass name after each optimization pass when the pass fails
    /// This is expected to be combined with `print_passes_after`.
    pub print_passes_on_failure: bool,
    #[option(default_value("false"))]
    /// Prints MLIR operations using their generic form
    pub mlir_print_generic_ops: bool,
    #[option(default_value("true"))]
    /// Prints the operation associated with an MLIR diagnostic
    pub mlir_print_op_on_diagnostic: bool,
    #[option(default_value("false"))]
    /// Prints the stacktrace associated with an MLIR diagnostic
    pub mlir_print_trace_on_diagnostic: bool,
    #[option(default_value("false"))]
    /// Prints MLIR at module scope when printing diagnostics
    pub mlir_print_module_scope: bool,
    #[option(default_value("false"))]
    /// Prints MLIR using local scope when printing diagnostics
    pub mlir_print_local_scope: bool,
    #[option(default_value("false"))]
    /// Enables verification of MLIR after each pass
    pub mlir_enable_verifier: bool,
    #[option(default_value("false"))]
    /// Enables timing instrumentation of MLIR passes
    pub mlir_enable_timing: bool,
    #[option(default_value("false"))]
    /// Enables statistics instrumentation of MLIR passes
    pub mlir_enable_statistics: bool,
    #[option(takes_value(true), value_name("PATH"))]
    /// Enables crash reproducer generation on MLIR pass failure to the given path
    pub mlir_enable_crash_reproducer: Option<PathBuf>,
    #[option(default_value("false"))]
    /// Prints the LLVM optimization passes being run
    pub print_llvm_passes: bool,
    #[option(default_value("false"))]
    /// Prints diagnostics for LLVM optimization remarks produced during codegen
    pub print_llvm_optimization_remarks: bool,
    #[option(
        takes_value(true),
        possible_values("full", "partial", "off", "none"),
        hidden(true)
    )]
    /// Choose which RELRO level to use")
    pub relro_level: Option<RelroLevel>,
    #[option(
        takes_value(true),
        possible_values("address", "leak", "memory", "thread"),
        hidden(true)
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
     * Tell the linker which information to strip:
     *     none      = do not strip anything
     *     debuginfo = strip debugging information
     *     symbols   = strip debugging symbols wh
     *     _
     */
    pub strip: Strip,
    #[option(hidden(true))]
    /// Enable ThinLTO when possible
    pub thinlto: Option<bool>,
    #[option(default_value("1"), takes_value(true), value_name("N"))]
    /// Use a thread pool with N threads
    pub threads: u64,
    #[option(hidden(true))]
    /// Measure time of each lumen pass
    pub time_passes: bool,
    #[option]
    /// Measure time of each LLVM pass
    pub time_mlir_passes: bool,
    #[option]
    /// Measure time of each LLVM pass
    pub time_llvm_passes: bool,
    #[option(default_value("true"))]
    /// Verify LLVM IR
    pub verify_llvm_ir: bool,
}
