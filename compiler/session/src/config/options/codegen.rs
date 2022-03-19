use std::path::PathBuf;

use liblumen_target::{CodeModel, LinkerFlavor, PanicStrategy, RelocModel, TlsModel};

use liblumen_compiler_macros::option_group;

use crate::config::*;

#[option_group(
    name = "codegen",
    short = "C",
    help = "Available flags for customizing code generation"
)]
#[derive(Debug, Clone, Default)]
pub struct CodegenOptions {
    #[option(value_name("MODEL"), takes_value(true), hidden(true))]
    /// Choose the code model to use
    pub code_model: Option<CodeModel>,
    #[option(
        next_line_help(true),
        takes_value(true),
        value_name("CHECK_TYPE"),
        default_value("disabled"),
        possible_values("false", "true", "disabled", "checks", "nochecks"),
        hidden(true)
    )]
    /**
     * Use Windows Control Flow Guard:
     *     checks   = emit metadata and checks
     *     nochecks = emit metadata but no checks
     *     disabled = do not emit metadata or checks
     *     true     = alias for `checks`
     *     false    = alias for `disabled`
     *     _
     */
    pub control_flow_guard: CFGuard,
    #[option]
    /// Enable debug assertions
    pub debug_assertions: Option<bool>,
    #[option(
        next_line_help(true),
        takes_value(true),
        value_name("LEVEL"),
        possible_values("0", "1", "2")
    )]
    /**
     * Debug info emission level
     *     0 = no debug info
     *     1 = line tables only,
     *     2 = full debug info with variable and type information
     *     _
     */
    pub debuginfo: Option<DebugInfo>,
    /**
     * Optimization level
     *     0 = no optimization
     *     1 = minimal optimizations
     *     2 = normal optimizations (default)
     *     3 = aggressive optimizations
     *     s = optimize for size
     *     z = aggressively optimize for size
     */
    #[option(
        next_line_help(true),
        takes_value(true),
        value_name("LEVEL"),
        possible_values("0", "1", "2", "3", "s", "z")
    )]
    pub opt_level: Option<OptLevel>,
    #[option(default_value("false"))]
    /// Allow the linker to link its default libraries
    pub default_linker_libraries: bool,
    #[option(default_value("false"), hidden(true))]
    pub embed_bitcode: bool,
    #[option(default_value("255"), value_name("N"), takes_value(true), hidden(true))]
    /// Set the threshold for inlining a function
    pub inline_threshold: Option<u64>,

    #[option(value_name("PATH"), takes_value(true))]
    /// The system linker to link with
    pub linker: Option<PathBuf>,
    #[option(multiple(true), takes_value(true), value_name("ARG"))]
    /// A single argument to append to the linker args (can be used multiple times)
    pub linker_arg: Vec<String>,
    #[option(value_name("ARGS"), takes_value(true), requires_delimiter(true))]
    /// Extra arguments to append to the linker invocation (comma separated list)
    pub linker_args: Option<Vec<String>>,
    #[option(value_name("FLAVOR"), takes_value(true))]
    /// Linker flavor, e.g. 'gcc', 'ld', 'msvc', 'wasm-ld'
    pub linker_flavor: Option<LinkerFlavor>,
    #[option(
        next_line_help(true),
        takes_value(true),
        default_value("false"),
        hidden(true)
    )]
    /**
     * Generate build artifacts that are compatible with linker-based LTO
     *     auto     = let the compiler choose
     *     disabled = do not build LTO-compatible artifacts (default)
     *     false    = alias for 'disabled'
     *     _
     */
    pub linker_plugin_lto: LinkerPluginLto,
    #[option]
    /// Prevent the linker from stripping dead code (useful for code coverage)
    pub link_dead_code: Option<bool>,
    #[option(value_name("ARGS"), takes_value(true), requires_delimiter(true))]
    /// Extra arguments to pass through to LLVM (comma separated list)
    pub llvm_args: Vec<String>,
    #[option(
        takes_value(true),
        hidden(true),
        possible_values("no", "yes", "thin", "fat")
    )]
    /// Perform link-time optimization
    pub lto: LtoCli,
    #[option(hidden(true))]
    /// Don't pre-populate the pass manager with a list of passes
    pub no_prepopulate_passes: bool,
    #[option(hidden(true))]
    /// When set, does not implicitly link the Lumen runtime
    pub no_std: Option<bool>,
    #[option(
        possible_values("abort", "unwind"),
        value_name("STRATEGY"),
        takes_value(true),
        hidden(true)
    )]
    /// Panic strategy to compile with
    pub panic: Option<PanicStrategy>,
    #[option(value_name("PASSES"), takes_value(true), requires_delimiter(true))]
    /// A list of extra LLVM passes to run (comma separated list)
    pub passes: Vec<String>,
    #[option]
    /// Prefer dynamic linking to static linking
    pub prefer_dynamic: bool,
    #[option(value_name("MODEL"), takes_value(true), hidden(true))]
    /// Choose the relocation model to use
    pub relocation_model: Option<RelocModel>,
    #[option(value_name("PASSES"), takes_value(true), hidden(true))]
    /// Print remarks for these optimization passes (comma separated, or 'all')
    pub remark: Passes,
    #[option]
    /// Set rpath values in libs/exes
    pub rpath: bool,
    #[option(hidden(true))]
    /// Run `dsymutil` and delete intermediate object files
    pub run_dsymutil: Option<bool>,
    #[option(value_name("CPU"), takes_value(true))]
    /// Select target processor (see `lumen print target-cpus`)
    pub target_cpu: Option<String>,
    #[option(value_name("FEATURES"), takes_value(true))]
    /// Select target specific attributes (see `lumen print target-features`)
    pub target_features: Option<String>,
    #[option(hidden(true))]
    /// Choose the TLS model to use
    pub tls_model: Option<TlsModel>,
}
