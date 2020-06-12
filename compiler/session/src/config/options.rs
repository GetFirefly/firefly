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

use clap::ArgMatches;

use liblumen_target::{self as target, Target};
use liblumen_util::diagnostics::{ColorArg, ColorChoice, FileName};
use liblumen_util::error::{HelpRequested, Verbosity};
use liblumen_util::fs::NativeLibraryKind;

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
    pub project_name: String,
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
    pub input_file: Option<FileName>,
    pub output_file: Option<PathBuf>,
    pub output_dir: Option<PathBuf>,
    // Remap source path prefixes in all output (messages, object files, debug, etc.).
    pub source_path_prefix: Vec<(PathBuf, PathBuf)>,
    pub search_paths: Vec<SearchPath>,
    pub include_path: VecDeque<PathBuf>,
    pub code_path: VecDeque<PathBuf>,
    pub link_libraries: Vec<(String, Option<String>, Option<NativeLibraryKind>)>,
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
        mut codegen_opts: CodegenOptions,
        debugging_opts: DebuggingOptions,
        cwd: PathBuf,
        args: &ArgMatches<'a>,
    ) -> Result<Self, anyhow::Error> {
        // Check the input argument first to see if help was requested
        let input_file = match args.value_of_os("input") {
            None => None,
            Some(input_arg) => match input_arg.to_str() {
                Some("help") => return Err(HelpRequested("compile", None).into()),
                Some("-") => Some("stdin".into()),
                Some(path) => Some(PathBuf::from(path).into()),
                None => Some(PathBuf::from(input_arg).into()),
            },
        };

        if let Some(extra_args) = args.values_of("raw") {
            for arg in extra_args {
                codegen_opts.llvm_args.push(arg.to_string());
            }
        }

        let project_name = detect_project_name(args, cwd.as_path(), input_file.as_ref());
        let project_type_opt: Option<ProjectType> =
            ParseOption::parse_option(&option!("project-type"), &args)?;
        let project_type = project_type_opt.unwrap_or(ProjectType::Executable);
        let output_types = OutputTypes::parse_option(&option!("emit"), &args)?;

        let color_arg = ColorArg::parse_option(&option!("color"), &args)?;

        let maybe_sysroot: Option<PathBuf> = ParseOption::parse_option(&option!("sysroot"), &args)?;
        let sysroot = match &maybe_sysroot {
            Some(sysroot) => sysroot.clone(),
            None => filesearch::get_or_default_sysroot(),
        };

        let target: Target = ParseOption::parse_option(&option!("target"), &args)?;
        match &target.target_pointer_width {
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

        let opt_level = if args.is_present("no-optimize") {
            OptLevel::No
        } else {
            ParseOption::parse_option(&option!("opt-level"), &args)?
        };

        let debug_info = if args.is_present("debug") {
            if codegen_opts.debuginfo.is_some() {
                // Prefer more precise option
                codegen_opts.debuginfo.clone().unwrap()
            } else {
                DebugInfo::Full
            }
        } else {
            codegen_opts.debuginfo.clone().unwrap_or(DebugInfo::None)
        };

        // The `-g` and `-C debuginfo` flags specify the same setting
        let debug_assertions = codegen_opts
            .debug_assertions
            .unwrap_or(opt_level == OptLevel::No);

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

        let output_file = args.value_of_os("output").map(PathBuf::from);
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
        let warnings_as_errors = args.is_present("warnings-as-errors");
        let no_warn = args.is_present("no-warn");
        let verbosity = Verbosity::from_level(args.occurrences_of("verbose") as isize);
        let include_path = VecDeque::new();
        let mut code_path = VecDeque::new();
        if let Some(values) = args.values_of_os("prepend-path") {
            for value in values {
                // Prepend in the order given
                code_path.push_front(PathBuf::from(value));
            }
        }
        if let Some(values) = args.values_of_os("append-path") {
            for value in values {
                // Append in the order given
                code_path.push_back(PathBuf::from(value));
            }
        }

        Ok(Self {
            project_name,
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
            input_file,
            output_file,
            output_dir,
            source_path_prefix,
            search_paths,
            include_path,
            code_path,
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
        codegen_opts: CodegenOptions,
        debugging_opts: DebuggingOptions,
        cwd: PathBuf,
        args: &ArgMatches<'a>,
    ) -> Result<Self, anyhow::Error> {
        let basename = cwd.file_name().unwrap();
        let project_name = basename.to_str().unwrap().to_owned();

        let target_opt: Option<Target> = ParseOption::parse_option(&option!("target"), &args)?;
        let target = target_opt.unwrap_or_else(|| {
            let triple = target::host_triple().to_string();
            Target::search(&triple).unwrap()
        });
        match &target.target_pointer_width {
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
            project_name,
            project_type: ProjectType::Executable,
            output_types: OutputTypes::default(),
            color: ColorChoice::Auto,
            warnings_as_errors: false,
            no_warn: false,
            verbosity: Verbosity::from_level(0),
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
            input_file: None,
            output_file: None,
            output_dir: None,
            source_path_prefix: vec![],
            search_paths: Default::default(),
            include_path: Default::default(),
            code_path: Default::default(),
            link_libraries: Default::default(),
            defines,
            cli_forced_thinlto_off: false,
        })
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
                cwd.push("lumen");
                cwd.push(self.target.triple());
                cwd
            })
    }

    pub fn crt_static(&self) -> bool {
        // If the target does not opt in to crt-static support, use its default.
        if self.target.options.crt_static_respected {
            self.crt_static_feature()
        } else {
            self.target.options.crt_static_default
        }
    }

    pub fn crt_static_feature(&self) -> bool {
        let requested_features = self
            .codegen_opts
            .target_features
            .as_ref()
            .map(|s| s.as_str())
            .unwrap_or_else(|| "")
            .split(',');
        let found_negative = requested_features.clone().any(|r| r == "-crt-static");
        let found_positive = requested_features.clone().any(|r| r == "+crt-static");

        // If the target we're compiling for requests a static crt by default,
        // then see if the `-crt-static` feature was passed to disable that.
        // Otherwise if we don't have a static crt by default then see if the
        // `+crt-static` feature was passed.
        if self.target.options.crt_static_default {
            !found_negative
        } else {
            found_positive
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
        !self.debugging_opts.parse_only // The file is just being parsed
    }
}

fn detect_project_name<'a>(args: &ArgMatches<'a>, cwd: &Path, input: Option<&FileName>) -> String {
    // If explicitly set, use the provided name
    if let Some(name) = args.value_of("name") {
        return name.to_owned();
    }
    match input {
        // If we have a single input file, name the project after it
        Some(FileName::Real(ref path)) if path.exists() && path.is_file() => path
            .file_stem()
            .unwrap()
            .to_str()
            .expect("invalid utf-8 in input file name")
            .to_owned(),
        // If we have an input directory, name the project after the directory
        Some(FileName::Real(ref path)) if path.exists() && path.is_dir() => path
            .file_name()
            .unwrap()
            .to_str()
            .expect("invalid utf-8 in input file name")
            .to_owned(),
        // Fallback to using the current working directory name
        _ => cwd.file_name().unwrap().to_str().unwrap().to_owned(),
    }
}

/// Generate a default project configuration for the current session
fn default_configuration(target: &Target) -> HashMap<String, Option<String>> {
    let end = target.target_endian.clone();
    let arch = target.arch.clone();
    let wordsz = target.target_pointer_width;
    let os = target.target_os.clone();
    let env = target.target_env.clone();
    let vendor = target.target_vendor.clone();

    let mut ret = HashMap::default();
    ret.reserve(6); // the minimum number of insertions
                    // Target bindings.
    ret.insert("TARGET_OS".to_string(), Some(os));
    if let Some(ref fam) = target.options.target_family {
        ret.insert("TARGET_FAMILY".to_string(), Some(fam.to_string()));
    }
    ret.insert("TARGET_POINTER_WIDTH".to_string(), Some(wordsz.to_string()));
    ret.insert("TARGET_ARCH".to_string(), Some(arch));
    ret.insert("TARGET_ENDIANESS".to_string(), Some(end.to_string()));
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
) -> Result<Vec<(String, Option<String>, Option<NativeLibraryKind>)>, clap::Error> {
    match matches.values_of("link-library") {
        None => return Ok(Vec::new()),
        Some(values) => {
            let mut link_libraries = Vec::new();
            for value in values {
                // Parse string of the form "[KIND=]lib[:new_name]",
                // where KIND is one of "dylib", "framework", "static".
                let mut parts = value.splitn(2, '=');
                let kind = parts.next().unwrap();
                let (name, kind) = match (parts.next(), kind) {
                    (None, name) => (name, None),
                    (Some(name), "dylib") => (name, Some(NativeLibraryKind::NativeUnknown)),
                    (Some(name), "framework") => (name, Some(NativeLibraryKind::NativeFramework)),
                    (Some(name), "static") => (name, Some(NativeLibraryKind::NativeStatic)),
                    (Some(name), "static-nobundle") => {
                        (name, Some(NativeLibraryKind::NativeStaticNobundle))
                    }
                    (_, s) => {
                        return Err(str_to_clap_err(
                            "link-library",
                            &format!(
                                "unknown library kind `{}`, expected \
                                 one of dylib, framework, or static",
                                s
                            ),
                        ))
                    }
                };
                let mut name_parts = name.splitn(2, ':');
                let name = name_parts.next().unwrap();
                let new_name = name_parts.next();
                link_libraries.push((name.to_owned(), new_name.map(|n| n.to_owned()), kind));
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
