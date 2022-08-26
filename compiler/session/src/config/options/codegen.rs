use std::path::PathBuf;

use firefly_target::{CodeModel, LinkerFlavor, RelocModel, TlsModel};
use firefly_target::{MergeFunctions, RelroLevel};

use firefly_compiler_macros::option_group;

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
    #[option(default_value("false"))]
    /// Allow the linker to link its default libraries
    pub default_linker_libraries: bool,
    #[option(default_value("false"), hidden(true))]
    pub embed_bitcode: bool,
    #[option(hidden(true))]
    pub force_frame_pointers: Option<bool>,
    #[option(hidden(true))]
    pub force_unwind_tables: Option<bool>,
    /// Set whether each function should go in its own section
    #[option(hidden(true))]
    pub function_sections: Option<bool>,
    #[option(hidden(true))]
    pub gcc_ld: Option<LdImpl>,
    #[option(value_name("N"), takes_value(true), hidden(true))]
    /// Set the threshold for inlining a function
    pub inline_threshold: Option<u64>,
    #[option(multiple(true), takes_value(true), value_name("ARG"))]
    /// A single argument to append to the linker args (can be used multiple times)
    pub linker_arg: Vec<String>,
    #[option(value_name("ARGS"), takes_value(true), requires_delimiter(true))]
    /// Extra arguments to append to the linker invocation (comma separated list)
    pub linker_args: Option<Vec<String>>,
    #[option]
    /// Prevent the linker from stripping dead code (useful for code coverage)
    pub link_dead_code: Option<bool>,
    #[option(default_value("true"))]
    /// Link native libraries in the linker invocation
    pub link_native_libraries: bool,
    #[option(hidden(true))]
    /// Control whether to link Rust provided C objects/libraries or rely on
    /// C toolchain installed on the system
    pub link_self_contained: Option<bool>,
    #[option(value_name("PATH"), takes_value(true))]
    /// The system linker to link with
    pub linker: Option<PathBuf>,
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
    /// Compile without linking
    #[option]
    pub no_link: bool,
    #[option(hidden(true))]
    /// Don't pre-populate the pass manager with a list of passes
    pub no_prepopulate_passes: bool,
    #[option(hidden(true))]
    /// When set, does not implicitly link the Firefly runtime
    pub no_std: Option<bool>,
    #[option(hidden(true))]
    pub no_unique_section_names: bool,
    /**
     * Optimization level
     *     0 = no optimization (default)
     *     1 = minimal optimizations
     *     2 = normal optimizations
     *     3 = aggressive optimizations
     *     s = optimize for size
     *     z = aggressively optimize for size
     */
    #[option(
        next_line_help(true),
        takes_value(true),
        value_name("LEVEL"),
        default_value("0"),
        possible_values("0", "1", "2", "3", "s", "z")
    )]
    pub opt_level: OptLevel,
    #[option]
    /// Pass `-install_name @rpath/...` to the macOS linker
    pub osx_rpath_install_name: bool,
    #[option(value_name("PASSES"), takes_value(true), requires_delimiter(true))]
    /// A list of extra LLVM passes to run (comma separated list)
    pub passes: Vec<String>,
    #[option(takes_value(true), value_name("ARG"))]
    /// A single extra argument to prepend the linker invocation
    /// can be used more than once
    pub pre_link_arg: Vec<String>,
    #[option(takes_value(true), value_name("ARGS"), require_delimiter(true))]
    /// Extra arguments to prepend to the linker invocation (space separated)
    pub pre_link_args: Option<Vec<String>>,
    #[option]
    /// Prefer dynamic linking to static linking
    pub prefer_dynamic: bool,
    #[option(value_name("MODEL"), takes_value(true), hidden(true))]
    /// Choose the relocation model to use
    pub relocation_model: Option<RelocModel>,
    #[option(
        takes_value(true),
        possible_values("full", "partial", "off", "none"),
        hidden(true)
    )]
    /// Choose which RELRO level to use")
    pub relro_level: Option<RelroLevel>,
    #[option(value_name("PASSES"), takes_value(true), hidden(true))]
    /// Print remarks for these optimization passes (comma separated, or 'all')
    pub remark: Passes,
    #[option]
    /// Set rpath values in libs/exes
    pub rpath: bool,
    /**
     * Tell the linker which information to strip:
     *     none      = do not strip anything
     *     debuginfo = strip debugging information
     *     symbols   = strip debugging symbols wh
     *     _
     */
    #[option(
        next_line_help(true),
        takes_value(true),
        value_name("TYPE"),
        default_value("none"),
        possible_values("none", "debuginfo", "symbols")
    )]
    pub strip: Strip,
    #[option(value_name("CPU"), takes_value(true))]
    /// Select target processor (see `firefly print target-cpus`)
    pub target_cpu: Option<String>,
    #[option(value_name("FEATURES"), takes_value(true))]
    /// Select target specific attributes (see `firefly print target-features`)
    pub target_features: Option<String>,
    #[option(hidden(true))]
    /// Enable ThinLTO when possible
    pub thinlto: Option<bool>,
    #[option(hidden(true))]
    /// Choose the TLS model to use
    pub tls_model: Option<TlsModel>,
    /// Whether to build a WASI command or reactor
    #[option(
        takes_value(true),
        value_name("MODEL"),
        possible_values("command", "reactor")
    )]
    pub wasi_exec_model: Option<WasiExecModel>,
}
