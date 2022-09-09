use std::ffi::OsString;

use clap::crate_description;
use clap::{App, AppSettings, Arg, ArgMatches};

use firefly_session::{CodegenOptions, DebuggingOptions, OptionGroup, OutputType};
use firefly_target::Target;
use firefly_util::diagnostics::ColorArg;

/// Parses the provided arguments
pub fn parse<'a>(args: impl Iterator<Item = OsString>) -> clap::Result<ArgMatches<'a>> {
    parser().get_matches_from_safe(args)
}

pub fn parser<'a, 'b>() -> App<'a, 'b> {
    App::new("firefly")
        .version(crate::FIREFLY_RELEASE)
        .about(crate_description!())
        .setting(AppSettings::UnifiedHelpMessage)
        .setting(AppSettings::GlobalVersion)
        .setting(AppSettings::VersionlessSubcommands)
        .setting(AppSettings::ArgRequiredElseHelp)
        .arg(
            CodegenOptions::option_group_arg()
                .global(true)
                .help("Set a codegen option. Use '-C help' for details"),
        )
        .arg(
            DebuggingOptions::option_group_arg()
                .global(true)
                .help("Set a debugging option. Use '-Z help' for details"),
        )
        .subcommand(print_command())
        .subcommand(compile_command())
}

/// Prints help for the given command
pub fn command_help(command: &str) {
    match command {
        "print" => print_command().print_help().unwrap(),
        "compile" => compile_command().print_help().unwrap(),
        other => {
            eprintln!("Help unavailable for '{}' command!", other);
        }
    }
}

fn print_command<'a, 'b>() -> App<'a, 'b> {
    let target = self::target_arg();
    App::new("print")
        .about("Prints compiler information to standard out")
        .setting(AppSettings::SubcommandRequired)
        .subcommand(
            App::new("version")
                .about("Prints version information for the compiler")
                .arg(
                    Arg::with_name("verbose")
                        .help("Print extra version details, such as commit hash")
                        .short("v")
                        .long("verbose"),
                ),
        )
        .subcommand(App::new("current-target").about("Prints details about the current target"))
        .subcommand(App::new("targets").about("The list of supported targets"))
        .subcommand(
            App::new("target-features")
                .about("Prints the available target features for the current target")
                .arg(target.clone().help("The target to list features for")),
        )
        .subcommand(
            App::new("target-cpus")
                .about("Prints the available architectures for the current target")
                .arg(target.clone().help("The target to list architectures for")),
        )
        .subcommand(
            App::new("passes").about("Prints the LLVM passes registered with the pass manager"),
        )
}

fn compile_command<'a, 'b>() -> App<'a, 'b> {
    let target = self::target_arg();
    App::new("compile")
        .about("Compiles Erlang sources to an executable or shared library")
        .setting(AppSettings::DeriveDisplayOrder)
        .arg(
            Arg::with_name("inputs")
                .index(1)
                .help(
                    "Path(s) to the source file(s) or director(y|ies) to compile.\n\
                     You may also use `-` as a file name to read a file from stdin.\n\
                     If not provided, the compiler will treat the current working directory\n\
                     as the root of a standard Erlang project, using sources from <cwd>/src.",
                )
                .next_line_help(true)
                .multiple(true)
                .value_name("INPUTS"),
        )
        .arg(
            Arg::with_name("bin")
                 .help("Tells the compiler to build this application into an executable (the default)")
                 .long("bin")
                 .conflicts_with("lib")
        )
        .arg(
            Arg::with_name("lib")
                 .help("Tells the compiler to build this application into a library")
                 .long("lib")
                 .conflicts_with("bin")
        )
        .arg(
            Arg::with_name("dynamic")
                 .help("When combined with --lib, builds this application as a shared library")
                 .long("dynamic")
                 .conflicts_with("bin")
                 .conflicts_with("static")
        )
        .arg(
            Arg::with_name("static")
                 .help("When combined with --lib, builds this application as a static library (the default)")
                 .long("static")
                 .conflicts_with("bin")
                 .conflicts_with("dynamic")
        )
        .arg(
            Arg::with_name("app-name")
                .help("Specify the name of the Erlang application being built")
                .long("app-name")
                .takes_value(true)
                .value_name("NAME"),
        )
        .arg(
            Arg::with_name("app-module")
                .help("Specify the name of the application callback module for this application")
                .long("app-module")
                .takes_value(true)
                .value_name("MODULE"),
        )
        .arg(
            Arg::with_name("app-version")
                 .help("Specify the version of the Erlang application being built")
                 .long("app-version")
                 .takes_value(true)
        )
        .arg(
            Arg::with_name("app")
                 .help("Path to the resource file (.app/.app.src) from which to read application metadata")
                 .long("app")
                 .takes_value(true)
                 .value_name("PATH")
                .conflicts_with("app-name")
                .conflicts_with("app-version")
                .conflicts_with("app-module")
        )
        .arg(
            Arg::with_name("output")
                .help("Write output to the given filename")
                .short("o")
                .long("output")
                .value_name("FILENAME"),
        )
        .arg(
            Arg::with_name("output-dir")
                .help("Write all outputs to DIR")
                .long("output-dir")
                .value_name("DIR"),
        )
        .arg(
            Arg::with_name("debug")
                .help("Generate source level debug information (same as -C debuginfo=2)")
                .short("g")
        )
        .arg(
            Arg::with_name("opt-level")
                .help("Optimize generated code (same as -C opt-level=2)")
                .short("O")
        )
        .arg(
            target
                .clone()
                .help("The target triple to compile against (e.g. x86_64-linux-gnu)"),
        )
        .arg(
            Arg::with_name("color")
                .help("Configure output colors")
                .long("color")
                .possible_values(ColorArg::VARIANTS)
                .case_insensitive(true)
        )
        .arg(
            Arg::with_name("source-map-prefix")
                .help("Remap source paths in all output (i.e. FROM/foo => TO/foo)")
                .long("source-map-prefix")
                .hidden(true)
                .takes_value(true)
                .value_name("FROM=TO"),
        )
        .arg(
            Arg::with_name("define")
                .help("Define a macro, e.g. -D TEST or -D FOO=BAR")
                .short("D")
                .long("define")
                .takes_value(true)
                .value_name("NAME[=VALUE]")
                .multiple(true)
                .number_of_values(1),
        )
        .arg(
            Arg::with_name("warn")
                .help(
                    "Modify how warnings are treated by the compiler.\n\
                     \n\
                     -Werror = treat all warnings as errors\n\
                     -W0     = disable warnings\n\
                     -Wall   = enable all warnings",
                )
                .next_line_help(true)
                .short("W")
                .long("warn")
                .takes_value(true)
                .value_name("LEVEL")
                .default_value("all"),
        )
        .arg(
            Arg::with_name("verbose")
                .help("Set verbosity level")
                .short("v")
                .multiple(true),
        )
        .arg(
            Arg::with_name("link-library")
                .help(
                    "Link the generated binary to the specified native library NAME.\n\
                     The optional KIND can be one of: static, dylib (default), or framework.\n\
                     \n\
                     Example: `firefly compile -lc ...` will link against the system libc",
                )
                .next_line_help(true)
                .short("l")
                .takes_value(true)
                .value_name("[KIND=]NAME")
                .multiple(true)
                .number_of_values(1),
        )
        .arg(
            Arg::with_name("search-path")
                .help(
                    "Add a directory to the library search path.\n\
                     The optional KIND can be one of: dependency, \
                     native, framework, or all (default)",
                )
                .next_line_help(true)
                .short("L")
                .takes_value(true)
                .value_name("[KIND=]PATH")
                .multiple(true)
                .number_of_values(1),
        )
        .arg(
            Arg::with_name("include-paths")
                .help("Add a path to the Erlang include path.")
                .long("include")
                .short("I")
                .value_name("PATH")
                .takes_value(true)
                .multiple(true)
                .number_of_values(1),
        )
        .arg(
            Arg::with_name("emit")
                .help(OutputType::help())
                .next_line_help(true)
                .long("emit")
                .takes_value(true)
                .value_name("TYPE[=GLOB],..")
                .multiple(true)
                .require_delimiter(true),
        )
}

fn target_arg<'a, 'b>() -> Arg<'a, 'b> {
    Arg::with_name("target")
        .short("t")
        .long("target")
        .takes_value(true)
        .value_name("TRIPLE")
        .validator(|triple| match Target::search(&triple) {
            Ok(_) => Ok(()),
            Err(err) => Err(err.to_string()),
        })
}
