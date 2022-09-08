use std::path::PathBuf;

use firefly_compiler_macros::option_group;
use firefly_target::SplitDebugInfo;

use crate::config::*;

#[option_group(name = "debugging", short = "Z", help = "Available debugging flags")]
#[derive(Debug, Clone, Default)]
pub struct DebuggingOptions {
    #[option]
    /// Generate comments into the assembly (may change behavior)
    pub asm_comments: bool,
    /**
     * Debug info emission level
     *     0 = no debug info
     *     1 = line tables only,
     *     2 = full debug info with variable and type information
     *     _
     */
    #[option(
        next_line_help(true),
        takes_value(true),
        value_name("LEVEL"),
        default_value("0"),
        possible_values("0", "1", "2")
    )]
    pub debuginfo: DebugInfo,
    #[option(hidden(true))]
    /// Emit a section containing stack size metadata
    pub emit_stack_sizes: bool,
    #[option(hidden(true))]
    /// Gather statistics about the input
    pub input_stats: bool,
    #[option]
    /// A list of LLVM plugins to enable (space separated)
    pub llvm_plugins: Vec<String>,
    #[option]
    pub llvm_time_trace: bool,
    #[option(takes_value(true), possible_values("false", "none", "plain", "pretty"))]
    /// Enable printing of debug info when printing MLIR
    pub mlir_print_debug_info: MlirDebugPrinting,
    #[option(hidden(true))]
    /// Print the arguments passed to the linker
    pub print_link_args: bool,
    #[option(default_value("false"))]
    /// Prints MLIR IR and pass name for each optimization pass before being run
    pub mlir_print_passes_before: bool,
    #[option(default_value("false"))]
    /// Prints MLIR IR and pass name for each optimization pass after being run
    pub mlir_print_passes_after: bool,
    #[option(default_value("false"))]
    /// Only print MLIR IR and pass name after each optimization pass when the IR changes
    /// This is expected to be combined with `print_passes_after`.
    pub mlir_print_passes_on_change: bool,
    #[option(default_value("false"))]
    /// Only print MLIR IR and pass name after each optimization pass when the pass fails
    /// This is expected to be combined with `print_passes_after`.
    pub mlir_print_passes_on_failure: bool,
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
    #[option]
    /// Parse only; do not compile, assemble, or link
    pub parse_only: bool,
    #[option]
    /// Parse and run semantic analysis only; do not compile, assemble, or link
    pub analyze_only: bool,
    #[option]
    pub print_artifact_sizes: bool,
    #[option]
    /// Prints the LLVM optimization passes being run
    pub print_llvm_passes: bool,
    #[option]
    /// Prints diagnostics for LLVM optimization remarks produced during codegen
    pub print_llvm_optimization_remarks: bool,
    /**
     *  A comma-separated list of sanitizers to enable:
     *      address = enable the address sanitizer
     *      leak    = enable the leak sanitizer
     *      memory  = enable the memory sanitizer
     *      thread  = enable the thread sanitizer
     */
    #[option(takes_value(true), value_name("SANITIZERS"), requires_delimiter(true))]
    pub sanitizers: Vec<Sanitizer>,
    /// Enable origins tracking in MemorySanitizer
    #[option]
    pub sanitizer_memory_track_origins: bool,
    #[option]
    pub split_debuginfo: Option<SplitDebugInfo>,
    /**
     * Split DWARF variant (only if -Csplit-debuginfo is enabled and relevant)
     *
     *     split  = sections which do not require reloation are split out and ignored (default)
     *     single = sections which do not require relocation are ignored
     **/
    #[option(
        takes_value(true),
        value_name("KIND"),
        default_value("split"),
        possible_values("split", "single")
    )]
    pub split_dwarf_kind: SplitDwarfKind,
    #[option]
    /// Provide minimal debug info in the object/executable to facilitate online
    /// symbolication/stack traces in the absence of .dwo/.dwp files when using split DWARF
    pub split_dwarf_inlining: bool,
    #[option(default_value("1"), takes_value(true), value_name("N"))]
    /// Use a thread pool with N threads
    pub threads: u64,
    /// Measure the time spent on tasks
    #[option]
    pub time: bool,
    #[option]
    /// Measure time of each Firefly pass
    pub time_passes: bool,
    #[option]
    /// Measure time of each LLVM pass
    pub time_mlir_passes: bool,
    #[option]
    /// Measure time of each LLVM pass
    pub time_llvm_passes: bool,
    #[option]
    /// Verify LLVM IR
    pub verify_llvm_ir: bool,
}
