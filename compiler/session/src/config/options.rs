mod codegen;
mod debugging;
mod option_group;
mod option_info;
mod parse;

pub use self::codegen::CodegenOptions;
pub use self::debugging::DebuggingOptions;
pub use self::option_group::{
    OptionGroup, OptionGroupParseResult, OptionGroupParser, ShowOptionGroupHelp,
};
pub use self::option_info::OptionInfo;
pub use self::parse::*;

use std::collections::HashMap;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::bail;
use clap::ArgMatches;

use firefly_intern::Symbol;
use firefly_target::spec::{CodeModel, RelocModel, SplitDebugInfo, TlsModel};
use firefly_target::{self as target, Target};
use firefly_util::diagnostics::*;
use firefly_util::error::{HelpRequested, Verbosity};
use firefly_util::fs::NativeLibraryKind;

use super::*;
use crate::filesearch;
use crate::search_paths::SearchPath;

/// Represents user-defined configuration, e.g. `-D KEY[=VALUE]`
#[derive(Clone, PartialEq)]
struct Define(String, Option<String>);
impl Define {
    pub fn name(&self) -> &str {
        self.0.as_str()
    }

    pub fn value(&self) -> Option<&str> {
        self.1.as_ref().map(|s| s.as_str())
    }
}

// The top-level command-line options struct.
#[derive(Clone, Debug)]
pub struct Options {
    /// This is the root application being compiled
    pub app: Arc<App>,
    /// This is the set of applications that are implicit/explicit dependencies of the root application
    pub dependencies: HashMap<Symbol, Arc<App>>,
    /// This is the type of project/output being compiled
    pub project_type: ProjectType,
    pub output_types: OutputTypes,
    pub color: ColorChoice,
    pub warnings_as_errors: bool,
    pub no_warn: bool,
    pub verbosity: Verbosity,

    pub host: Target,
    pub target: Target,
    pub opt_level: OptLevel,
    pub debug_info: DebugInfo,
    pub debug_assertions: bool,
    pub test: bool,
    pub sysroot: PathBuf,
    pub host_tlib_path: SearchPath,
    /// `None` if the host and target are the same.
    pub target_tlib_path: Option<SearchPath>,

    pub codegen_opts: CodegenOptions,
    pub debugging_opts: DebuggingOptions,

    pub current_dir: PathBuf,
    /// This is the set of inputs per application
    pub input_files: HashMap<Symbol, Vec<FileName>>,
    pub output_file: Option<PathBuf>,
    pub output_dir: Option<PathBuf>,
    // Remap source path prefixes in all output (messages, object files, debug, etc.).
    pub source_path_prefix: Vec<(PathBuf, PathBuf)>,
    pub search_paths: Vec<SearchPath>,
    pub include_path: VecDeque<PathBuf>,
    pub link_libraries: Vec<(String, Option<String>, NativeLibraryKind)>,
    pub defines: HashMap<String, Option<String>>,

    pub cli_forced_thinlto_off: bool,
}

macro_rules! option {
    ($name:expr) => {
        OptionInfo::from_name($name)
    };
}

impl Options {
    pub fn new<'a>(
        diagnostics: Option<Arc<DiagnosticsHandler>>,
        codemap: Arc<CodeMap>,
        codegen_opts: CodegenOptions,
        debugging_opts: DebuggingOptions,
        cwd: PathBuf,
        args: &ArgMatches<'a>,
    ) -> anyhow::Result<Self> {
        let (warnings_as_errors, no_warn) = match args.value_of("warn") {
            Some("0") | Some("none") => (false, true),
            Some("error") => (true, false),
            None | Some(_) => (false, false),
        };
        let verbosity = Verbosity::from_level(args.occurrences_of("verbose") as isize);

        // Create a temporary diagnostics handler
        let diagnostics = diagnostics.unwrap_or_else(|| {
            create_diagnostics_handler(None, codemap.clone(), Verbosity::Info, false, false)
        });

        let output_file = args.value_of_os("output").map(PathBuf::from);
        let (app, dependencies, input_files) = match args.values_of_os("inputs") {
            None => {
                // By default treat the current working directory as a standard Erlang app
                let inputs = vec![FileName::Real(cwd.clone())];
                let app = detect_app(
                    &diagnostics,
                    codemap,
                    args,
                    cwd.as_path(),
                    output_file.as_deref(),
                    inputs.as_slice(),
                )?;
                let mut input_files = HashMap::new();
                input_files.insert(app.name, inputs);
                (app, HashMap::default(), input_files)
            }
            Some(mut input_args) => {
                let num_inputs = input_args.len();
                let first_os = input_args.next().unwrap();
                let first = first_os.to_str();

                // Check the input argument first to see if help was requested
                match first {
                    Some("help") => return Err(HelpRequested("compile", None).into()),
                    Some("-") => {
                        if num_inputs > 1 {
                            bail!("stdin as an input cannot be combined with other inputs");
                        }
                        let inputs = vec!["stdin".into()];
                        let app = detect_app(
                            &diagnostics,
                            codemap,
                            args,
                            cwd.as_path(),
                            output_file.as_deref(),
                            inputs.as_slice(),
                        )?;
                        let mut input_files = HashMap::new();
                        input_files.insert(app.name, inputs);
                        (app, HashMap::default(), input_files)
                    }
                    other => {
                        // Each directory is its own app, all assorted source files are associated to the root app
                        let first_filename: FileName = other
                            .map(PathBuf::from)
                            .unwrap_or_else(|| PathBuf::from(first_os))
                            .into();
                        let mut dirs = vec![];
                        let mut files = vec![];
                        if first_filename.is_dir() {
                            dirs.push(first_filename);
                        } else {
                            files.push(first_filename);
                        }

                        for input_arg in input_args {
                            let path = PathBuf::from(input_arg);
                            if !path.exists() {
                                bail!(
                                    "invalid input path, no such file or directory: {}",
                                    path.display()
                                );
                            }
                            if path.is_dir() {
                                dirs.push(path.into());
                            } else {
                                files.push(path.into());
                            }
                        }

                        let mut dependencies = HashMap::new();
                        let mut input_files = HashMap::new();

                        if files.is_empty() {
                            // We have only directories to process, create an app for each one and register the inputs
                            // If we have only a single directory, treat it as the root application
                            if dirs.len() == 1 {
                                let app = detect_app(
                                    &diagnostics,
                                    codemap,
                                    args,
                                    cwd.as_path(),
                                    output_file.as_deref(),
                                    dirs.as_slice(),
                                )?;
                                input_files.insert(app.name, dirs);
                                (app, dependencies, input_files)
                            } else {
                                // Don't use the root args when handling dependencies
                                let empty_args = ArgMatches::default();
                                for appdir in dirs.drain(..) {
                                    let app = detect_app(
                                        &diagnostics,
                                        codemap.clone(),
                                        &empty_args,
                                        cwd.as_path(),
                                        output_file.as_deref(),
                                        core::slice::from_ref(&appdir),
                                    )?;
                                    input_files.insert(app.name, vec![appdir]);
                                    dependencies.insert(app.name, app);
                                }
                                let app = detect_app(
                                    &diagnostics,
                                    codemap,
                                    args,
                                    cwd.as_path(),
                                    output_file.as_deref(),
                                    &[],
                                )?;
                                input_files.insert(app.name, vec![]);
                                (app, dependencies, input_files)
                            }
                        } else if dirs.is_empty() {
                            // We have only files to process, which all belong to the root app
                            let app = detect_app(
                                &diagnostics,
                                codemap,
                                args,
                                cwd.as_path(),
                                output_file.as_deref(),
                                files.as_slice(),
                            )?;
                            input_files.insert(app.name, files);
                            (app, dependencies, input_files)
                        } else {
                            // We have a mix of files and directories
                            let empty_args = ArgMatches::default();
                            for appdir in dirs.drain(..) {
                                let app = detect_app(
                                    &diagnostics,
                                    codemap.clone(),
                                    &empty_args,
                                    cwd.as_path(),
                                    output_file.as_deref(),
                                    core::slice::from_ref(&appdir),
                                )?;
                                input_files.insert(app.name, vec![appdir]);
                                dependencies.insert(app.name, app);
                            }
                            let app = detect_app(
                                &diagnostics,
                                codemap,
                                args,
                                cwd.as_path(),
                                output_file.as_deref(),
                                files.as_slice(),
                            )?;
                            input_files.insert(app.name, files);
                            (app, dependencies, input_files)
                        }
                    }
                }
            }
        };

        let project_type = if args.is_present("lib") {
            if args.is_present("dynamic") {
                ProjectType::Dylib
            } else {
                ProjectType::Staticlib
            }
        } else {
            ProjectType::Executable
        };
        let output_types = OutputTypes::parse_option(&option!("emit"), &args)?;
        let color_arg = ColorArg::parse_option(&option!("color"), &args)?;

        let maybe_sysroot: Option<PathBuf> = ParseOption::parse_option(&option!("sysroot"), &args)?;
        let sysroot = match &maybe_sysroot {
            Some(sysroot) => sysroot.clone(),
            None => filesearch::get_or_default_sysroot(),
        };

        let target: Target = ParseOption::parse_option(&option!("target"), &args)?;
        match &target.pointer_width {
            32 | 64 => (),
            w => {
                return Err(str_to_clap_err(
                    "target",
                    &format!(
                        "Invalid target: target pointer width of {} is not supported",
                        w
                    ),
                )
                .into())
            }
        }
        let host_triple = target::host_triple();
        let host = Target::search(host_triple)?;
        let target_triple = target.triple();
        let host_tlib_path = SearchPath::from_sysroot_and_triple(&sysroot, host_triple);
        let target_tlib_path = if host_triple == target_triple {
            None
        } else {
            Some(SearchPath::from_sysroot_and_triple(&sysroot, target_triple))
        };

        let mut defines = default_configuration(&target);

        let opt_level = if args.is_present("opt-level") {
            ParseOption::parse_option(&option!("opt-level"), &args)?
        } else {
            codegen_opts.opt_level
        };

        let debug_info = if args.is_present("debug") {
            ParseOption::parse_option(&option!("debug"), &args)?
        } else {
            debugging_opts.debuginfo
        };

        let debug_assertions = codegen_opts.debug_assertions.unwrap_or(false);

        if debug_assertions {
            defines.insert("DEBUG".to_string(), None);
        }

        let mut search_paths = vec![];
        match args.values_of("search-path") {
            None => (),
            Some(values) => {
                for value in values {
                    search_paths.push(
                        SearchPath::from_cli_opt(value)
                            .map_err(|e| str_to_clap_err("search-path", e))?,
                    );
                }
            }
        }

        let link_libraries = parse_link_libraries(&args)?;
        let source_path_prefix = parse_source_path_prefix(&args)?;

        let output_dir = args.value_of_os("output-dir").map(PathBuf::from);
        if let Some(values) = args.values_of("define") {
            for value in values {
                let define = self::parse_key_value(value)?;
                defines.insert(
                    define.name().to_string(),
                    define.value().map(|s| s.to_string()),
                );
            }
        }
        let mut include_path = VecDeque::new();
        let local_include_path = cwd.join("include");
        if local_include_path.exists() && local_include_path.is_dir() {
            include_path.push_front(local_include_path);
        }
        if let Some(values) = args.values_of_os("include-paths") {
            for value in values {
                include_path.push_front(PathBuf::from(value));
            }
        }

        Ok(Self {
            app,
            dependencies,
            project_type,
            output_types,
            color: color_arg.into(),
            warnings_as_errors,
            no_warn,
            verbosity,
            host,
            target,
            opt_level,
            debug_info,
            debug_assertions,
            test: false,
            sysroot,
            host_tlib_path,
            target_tlib_path,
            codegen_opts,
            debugging_opts,
            current_dir: cwd,
            input_files,
            output_file,
            output_dir,
            source_path_prefix,
            search_paths,
            include_path,
            link_libraries,
            defines,
            cli_forced_thinlto_off: false,
        })
    }

    // Don't try to parse all arguments, just backfill with defaults
    //
    // This is used by commands which need `Options`, but don't take
    // the full set of flags/options required by the compiler for actual
    // compilation
    pub fn new_with_defaults<'a>(
        diagnostics: Option<Arc<DiagnosticsHandler>>,
        codemap: Arc<CodeMap>,
        codegen_opts: CodegenOptions,
        debugging_opts: DebuggingOptions,
        cwd: PathBuf,
        args: &ArgMatches<'a>,
    ) -> anyhow::Result<Self> {
        let diagnostics = diagnostics.unwrap_or_else(|| {
            create_diagnostics_handler(None, codemap.clone(), Verbosity::Info, false, false)
        });
        let output_file = args.value_of_os("output").map(PathBuf::from);
        let app = detect_app(
            &diagnostics,
            codemap,
            args,
            &cwd,
            output_file.as_deref(),
            &[],
        )?;

        let target_opt: Option<Target> = ParseOption::parse_option(&option!("target"), &args)?;
        let target = target_opt.unwrap_or_else(|| {
            let triple = target::host_triple().to_string();
            Target::search(&triple).unwrap()
        });
        match &target.pointer_width {
            32 | 64 => (),
            w => {
                return Err(str_to_clap_err(
                    "target",
                    &format!(
                        "Invalid target: target pointer width of {} is not supported",
                        w
                    ),
                )
                .into())
            }
        }

        let defines = default_configuration(&target);
        let sysroot = filesearch::get_or_default_sysroot();
        let host_triple = target::host_triple();
        let host = Target::search(host_triple)?;
        let target_triple = target.triple();
        let host_tlib_path = SearchPath::from_sysroot_and_triple(&sysroot, host_triple);
        let target_tlib_path = if host_triple == target_triple {
            None
        } else {
            Some(SearchPath::from_sysroot_and_triple(&sysroot, target_triple))
        };

        Ok(Self {
            app,
            dependencies: HashMap::default(),
            project_type: ProjectType::Executable,
            output_types: OutputTypes::default(),
            color: ColorChoice::Auto,
            warnings_as_errors: false,
            no_warn: false,
            verbosity: Verbosity::Info,
            host,
            target,
            opt_level: OptLevel::Default,
            debug_info: DebugInfo::None,
            debug_assertions: false,
            test: false,
            sysroot,
            host_tlib_path,
            target_tlib_path,
            codegen_opts,
            debugging_opts,
            current_dir: cwd,
            input_files: HashMap::default(),
            output_file: None,
            output_dir: None,
            source_path_prefix: vec![],
            search_paths: Default::default(),
            include_path: Default::default(),
            link_libraries: Default::default(),
            defines,
            cli_forced_thinlto_off: false,
        })
    }

    /// Determines whether we should invoke the linker for the program being compiled
    pub fn should_link(&self) -> bool {
        !self.debugging_opts.parse_only
            && !self.debugging_opts.analyze_only
            && self.output_types.contains_key(&OutputType::Link)
    }

    pub fn maybe_emit(&self, input: &Input, output_type: OutputType) -> Option<PathBuf> {
        if self.debugging_opts.parse_only || self.debugging_opts.analyze_only {
            return None;
        }

        self.output_types
            .maybe_emit(input, output_type)
            .map(|p| match self.output_dir.as_ref() {
                Some(base) => base.join(p),
                None => p,
            })
    }

    pub fn maybe_emit_bytecode(&self) -> Option<PathBuf> {
        if self.output_types.contains_key(&OutputType::Bytecode) {
            Some(output::output_filename(
                self.output_file
                    .as_deref()
                    .and_then(|p| p.file_stem().map(PathBuf::from))
                    .map(FileName::real)
                    .unwrap_or_else(|| FileName::real(self.app.name.as_str().get())),
                OutputType::Bytecode,
                self.output_dir.as_deref(),
            ))
        } else {
            None
        }
    }

    pub fn lto(&self) -> Lto {
        match self.codegen_opts.lto {
            LtoCli::No => Lto::No,
            LtoCli::Yes => Lto::Fat,
            LtoCli::Thin => Lto::Thin,
            LtoCli::Fat => Lto::Fat,
            LtoCli::Unspecified => Lto::No,
        }
    }

    pub fn output_dir(&self) -> PathBuf {
        self.output_dir
            .as_ref()
            .map(|p| p.clone())
            .unwrap_or_else(|| {
                let mut cwd = self.current_dir.clone();
                cwd.push("_build");
                cwd.push("firefly");
                cwd.push(self.target.triple());
                cwd
            })
    }

    pub fn relocation_model(&self) -> RelocModel {
        self.codegen_opts
            .relocation_model
            .unwrap_or(self.target.options.relocation_model)
    }

    pub fn code_model(&self) -> Option<CodeModel> {
        self.codegen_opts
            .code_model
            .or(self.target.options.code_model)
    }

    pub fn tls_model(&self) -> TlsModel {
        self.codegen_opts
            .tls_model
            .unwrap_or(self.target.options.tls_model)
    }

    pub fn is_wasi_reactor(&self) -> bool {
        self.target.options.os == "wasi"
            && self.codegen_opts.wasi_exec_model == Some(WasiExecModel::Reactor)
    }

    pub fn split_debuginfo(&self) -> SplitDebugInfo {
        self.debugging_opts
            .split_debuginfo
            .unwrap_or(self.target.options.split_debuginfo)
    }

    pub fn target_can_use_split_dwarf(&self) -> bool {
        !self.target.options.is_like_windows && !self.target.options.is_like_osx
    }

    /// Check whether this compile session and crate type use static crt.
    pub fn crt_static(&self, _app_type: Option<ProjectType>) -> bool {
        if !self.target.options.crt_static_respected {
            // If the target does not opt in to crt-static support, use its default.
            return self.target.options.crt_static_default;
        }

        if let Some(ref requested_features) = self.codegen_opts.target_features {
            let features = requested_features.split(',');
            let found_negative = features.clone().any(|r| r == "-crt-static");
            let found_positive = features.clone().any(|r| r == "+crt-static");

            if found_positive || found_negative {
                found_positive
            } else {
                self.target.options.crt_static_default
            }
        } else {
            self.target.options.crt_static_default
        }
    }

    pub fn target_filesearch(
        &self,
        kind: crate::search_paths::PathKind,
    ) -> filesearch::FileSearch<'_> {
        filesearch::FileSearch::new(
            &self.sysroot,
            self.target.triple(),
            &self.search_paths,
            // `target_tlib_path == None` means it's the same as `host_tlib_path`.
            self.target_tlib_path
                .as_ref()
                .unwrap_or(&self.host_tlib_path),
            kind,
        )
    }

    pub fn host_filesearch(
        &self,
        kind: crate::search_paths::PathKind,
    ) -> filesearch::FileSearch<'_> {
        filesearch::FileSearch::new(
            &self.sysroot,
            target::host_triple(),
            &self.search_paths,
            &self.host_tlib_path,
            kind,
        )
    }

    pub fn set_host_target(&mut self, target: Target) {
        self.host = target;
    }

    /// Returns `true` if there will be an output file generated.
    pub fn will_create_output_file(&self) -> bool {
        !(self.debugging_opts.parse_only || self.debugging_opts.analyze_only) // The file is just being parsed/analyzed
    }

    /// Returns a list of directories where target-specific tool binaries are located.
    pub fn get_tools_search_paths(&self, self_contained: bool) -> Vec<PathBuf> {
        filesearch::get_tools_search_paths(&self.sysroot, self_contained)
    }

    pub fn exported_symbols(&self) -> Vec<String> {
        if let Some(ref exports) = self.target.options.override_export_symbols {
            return exports.iter().map(ToString::to_string).collect();
        }

        vec![]
    }

    #[inline]
    pub fn default_emitter(&self) -> Arc<dyn Emitter> {
        default_emitter(self.verbosity, self.color)
    }

    pub fn create_diagnostics_handler(
        &self,
        codemap: Arc<CodeMap>,
        emitter: Option<Arc<dyn Emitter>>,
    ) -> Arc<DiagnosticsHandler> {
        create_diagnostics_handler(
            emitter,
            codemap,
            self.verbosity,
            self.warnings_as_errors,
            self.no_warn,
        )
    }
}

fn default_emitter(verbosity: Verbosity, color: ColorChoice) -> Arc<dyn Emitter> {
    match verbosity {
        Verbosity::Silent => Arc::new(NullEmitter::new(color)),
        _ => Arc::new(DefaultEmitter::new(color)),
    }
}

fn create_diagnostics_handler(
    emitter: Option<Arc<dyn Emitter>>,
    codemap: Arc<CodeMap>,
    verbosity: Verbosity,
    warnings_as_errors: bool,
    no_warn: bool,
) -> Arc<DiagnosticsHandler> {
    Arc::new(DiagnosticsHandler::new(
        DiagnosticsConfig {
            verbosity,
            warnings_as_errors,
            no_warn,
            display: DisplayConfig::default(),
        },
        codemap.clone(),
        emitter.unwrap_or_else(|| default_emitter(verbosity, ColorChoice::Auto)),
    ))
}

/// Fetch the application metadata for the given src directory
///
/// If there is no .app/.app.src file in the given directory, returns Ok(None).
/// If there is a .app/.app.src file in the given directory, but it is invalid, returns Err.
/// Otherwise returns Ok(Some(app))
///
/// NOTE: Assumes that srcdir exists, will panic otherwise
fn try_load_app<'a>(
    diagnostics: &DiagnosticsHandler,
    codemap: Arc<CodeMap>,
    srcdir: &Path,
) -> anyhow::Result<Option<Arc<App>>> {
    // Locate the default .app file for the given src directory
    let default_appsrc = {
        srcdir.read_dir().unwrap().find_map(|entry| {
            if let Ok(entry) = entry {
                let path = entry.path();
                if !path.is_file() {
                    return None;
                }
                if path.ends_with(".app") || path.ends_with(".app.src") {
                    return Some(path.canonicalize().unwrap());
                }
            }

            None
        })
    };

    if let Some(path) = default_appsrc {
        Ok(Some(App::parse(&diagnostics, codemap, &path)?))
    } else {
        Ok(None)
    }
}

/// Fetch or generate application metadata based on the provided inputs
fn detect_app<'a>(
    diagnostics: &DiagnosticsHandler,
    codemap: Arc<CodeMap>,
    args: &ArgMatches<'a>,
    cwd: &Path,
    output_file: Option<&Path>,
    input_file_names: &[FileName],
) -> anyhow::Result<Arc<App>> {
    // If an path was explicitly provided, always prefer it
    if let Some(appsrc) = args.value_of("app") {
        let path = Path::new(appsrc);
        if !path.exists() || !path.is_file() {
            bail!("invalid application resource file: {}", path.display());
        }
        return Ok(App::parse(&diagnostics, codemap, path)?);
    }

    // For the remaining variations, the version is always handled the same
    let version = args.value_of("app-version").map(|v| v.to_string());

    // If the application module was manually specified, use it
    let otp_module = args.value_of("app-module").map(Symbol::intern);

    // If the application name was manually specified, use it
    if let Some(name) = args.value_of("app-name") {
        let name = Symbol::intern(name);
        let mut app = App::new(name);
        app.version = version;
        app.otp_module = otp_module;
        return Ok(Arc::new(app));
    }

    // We're left with inferring the application metadata.
    //
    // If an output filename was given, use the basename of that file as the app name.
    // Otherwise, if we have a single input, and it's a:
    //
    // * directory: try to treat that directory as a standard Erlang application
    // * file: use the file name as the name of the application
    //
    // Otherwise, if we have multiple inputs we use the name
    // of the current working directory as the application name.
    if let Some(output_file) = output_file {
        let name = Symbol::intern(output_file.file_stem().unwrap().to_str().unwrap());
        let mut app = App::new(name);
        app.version = version;
        app.otp_module = otp_module;
        return Ok(Arc::new(app));
    }

    if input_file_names.len() == 1 {
        let input = &input_file_names[0];
        if input.is_dir() {
            let input_dir: &Path = input.as_ref();
            let srcdir = input_dir.join("src");
            if let Ok(Some(app)) = try_load_app(&diagnostics, codemap.clone(), &srcdir) {
                return Ok(app);
            }
            if let Ok(Some(app)) = try_load_app(&diagnostics, codemap, input_dir) {
                return Ok(app);
            }
        }
        let name = match input {
            FileName::Real(ref path) => Symbol::intern(path.file_stem().unwrap().to_str().unwrap()),
            FileName::Virtual(ref name) => Symbol::intern(name.as_ref()),
        };
        let mut app = App::new(name);
        app.version = version;
        app.otp_module = otp_module;
        Ok(Arc::new(app))
    } else {
        let name = Symbol::intern(cwd.file_name().unwrap().to_str().unwrap());
        let mut app = App::new(name);
        app.version = version;
        app.otp_module = otp_module;
        Ok(Arc::new(app))
    }
}

/// Generate a default project configuration for the current session
fn default_configuration(target: &Target) -> HashMap<String, Option<String>> {
    let end = target.options.endianness.to_string();
    let arch = target.arch.to_string();
    let wordsz = target.pointer_width.to_string();
    let os = target.options.os.to_string();
    let env = target.options.env.to_string();
    let vendor = target.options.vendor.to_string();

    let mut ret = HashMap::default();
    ret.reserve(6); // the minimum number of insertions
                    // Target bindings.
    ret.insert("TARGET_OS".to_string(), Some(os));
    ret.insert("TARGET_POINTER_WIDTH".to_string(), Some(wordsz));
    ret.insert("TARGET_ARCH".to_string(), Some(arch));
    ret.insert("TARGET_ENDIANESS".to_string(), Some(end));
    ret.insert("TARGET_ENV".to_string(), Some(env));
    ret.insert("TARGET_VENDOR".to_string(), Some(vendor));
    ret
}

fn parse_key_value(value: &str) -> Result<Define, clap::Error> {
    let kv = value.splitn(2, '=').collect::<Vec<_>>();
    let key = kv[0].to_string();
    if kv.len() == 1 {
        return Ok(Define(key, None));
    }
    let value = kv[1].to_string();
    Ok(Define(key, Some(value)))
}

fn parse_link_libraries<'a>(
    matches: &ArgMatches<'a>,
) -> Result<Vec<(String, Option<String>, NativeLibraryKind)>, clap::Error> {
    match matches.values_of("link-library") {
        None => return Ok(Vec::new()),
        Some(values) => {
            let mut link_libraries = Vec::new();
            for value in values {
                // Parse string of the form "[KIND=]lib[:new_name]",
                //
                // * KIND is of the form `TYPE[:([+-]MODIFIER,?)*]
                // * TYPE is one of "dylib", "framework", "static".
                // * MODIFIER is one of "bundle", "as-needed", "whole-archive"
                //
                // When a modifier is `-`, it implies false; when `+`, it implies true
                let mut parts = value.splitn(2, '=');
                let kind = parts.next().unwrap();
                let (name, kind) = match (parts.next(), kind) {
                    (None, name) => (name, None),
                    (Some(name), kind) => (name, Some(kind)),
                };
                let (kind, modifiers) = match kind {
                    None => (None, vec![]),
                    Some(kind) => match kind.split_once(':') {
                        None => (Some(kind), vec![]),
                        Some((kind, modifiers)) => {
                            let mods = modifiers
                                .split(',')
                                .map(|m| match m.split_at(1) {
                                    ("-", m) => Some((m, false)),
                                    ("+", m) => Some((m, true)),
                                    _ => None,
                                })
                                .filter_map(|m| m)
                                .collect::<Vec<_>>();
                            (Some(kind), mods)
                        }
                    },
                };
                let lib = match (kind, modifiers) {
                    (None, _mods) => NativeLibraryKind::Unspecified,
                    (Some("dylib"), mods) => {
                        let as_needed =
                            mods.iter().find_map(
                                |(m, val)| if m == &"as-needed" { Some(*val) } else { None },
                            );
                        NativeLibraryKind::Dylib { as_needed }
                    }
                    (Some("framework"), mods) => {
                        let as_needed =
                            mods.iter().find_map(
                                |(m, val)| if m == &"as-needed" { Some(*val) } else { None },
                            );
                        NativeLibraryKind::Framework { as_needed }
                    }
                    (Some("static"), mods) => {
                        let bundle =
                            mods.iter().find_map(
                                |(m, val)| if m == &"bundle" { Some(*val) } else { None },
                            );
                        let whole_archive = mods.iter().find_map(|(m, val)| {
                            if m == &"whole-archive" {
                                Some(*val)
                            } else {
                                None
                            }
                        });
                        NativeLibraryKind::Static {
                            bundle,
                            whole_archive,
                        }
                    }
                    (Some("static-nobundle"), mods) => {
                        let whole_archive = mods.iter().find_map(|(m, val)| {
                            if m == &"whole-archive" {
                                Some(*val)
                            } else {
                                None
                            }
                        });
                        NativeLibraryKind::Static {
                            bundle: Some(false),
                            whole_archive,
                        }
                    }
                    (Some(other), _mods) => {
                        return Err(str_to_clap_err(
                            "link-library",
                            &format!(
                                "unknown library kind `{}`, expected \
                                 one of static, dylib, or framework",
                                other
                            ),
                        ))
                    }
                };
                let mut name_parts = name.splitn(2, ':');
                let name = name_parts.next().unwrap();
                let new_name = name_parts.next();
                link_libraries.push((name.to_owned(), new_name.map(|n| n.to_owned()), lib));
            }
            Ok(link_libraries)
        }
    }
}

fn parse_source_path_prefix<'a>(
    matches: &ArgMatches<'a>,
) -> Result<Vec<(PathBuf, PathBuf)>, clap::Error> {
    match matches.values_of("source-path-prefix") {
        None => return Ok(Vec::new()),
        Some(values) => {
            let mut source_maps = Vec::new();
            for value in values {
                let mut parts = value.rsplitn(2, '='); // reverse iterator
                let to = parts.next();
                let from = parts.next();
                match (from, to) {
                    (Some(from), Some(to)) => {
                        source_maps.push((PathBuf::from(from), PathBuf::from(to)))
                    }
                    _ => {
                        return Err(str_to_clap_err(
                            "source-path-prefix",
                            "invalid argument format, expected `PATH=PREFIX`",
                        ))
                    }
                }
            }
            Ok(source_maps)
        }
    }
}

pub fn str_to_clap_err(opt: &str, err: &str) -> clap::Error {
    clap::Error {
        kind: clap::ErrorKind::InvalidValue,
        message: err.to_string(),
        info: Some(vec![opt.to_string()]),
    }
}
