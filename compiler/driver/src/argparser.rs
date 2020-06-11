use std::ffi::OsString;

use clap::crate_description;
use clap::{App, AppSettings, Arg, ArgMatches};

use liblumen_session::{CodegenOptions, DebuggingOptions, OptionGroup, OutputType};
use liblumen_target::Target;
use liblumen_util::diagnostics::ColorArg;

/// Parses the provided arguments
pub fn parse<'a>(args: impl Iterator<Item = OsString>) -> clap::Result<ArgMatches<'a>> {
    parser().get_matches_from_safe(args)
}

pub fn parser<'a, 'b>() -> App<'a, 'b> {
    App::new("lumen")
        .version(crate::LUMEN_RELEASE)
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

pub fn print_print_help() {
    print_command().print_help().expect("unable to print help");
}

pub fn print_compile_help() {
    compile_command()
        .print_help()
        .expect("unable to print help");
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
        .subcommand(App::new("project-name").about("Prints the current project name"))
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
            Arg::with_name("input")
                .index(1)
                .help(
                    "Path to the source file or directory to compile.\n\
                     You may also use `-` as a file name to read a file from stdin.\n\
                     If not provided, the compiler will use the current directory as input.",
                )
                .next_line_help(true)
                .takes_value(true)
                .value_name("PATH"),
        )
        .arg(
            Arg::with_name("raw")
                .last(true)
                .help(
                    "Extra arguments that will be passed unmodified to the LLVM argument processor",
                )
                .next_line_help(true)
                .multiple(true)
                .value_name("ARGS"),
        )
        .arg(
            Arg::with_name("name")
                .help("Specify the name of the project being built")
                .short("n")
                .long("name")
                .takes_value(true)
                .value_name("NAME"),
        )
        .arg(
            Arg::with_name("output")
                .help("Write output to FILE")
                .long("output")
                .short("o")
                .value_name("FILE"),
        )
        .arg(
            Arg::with_name("output-dir")
                .help("Write output to file(s) in DIR")
                .long("output-dir")
                .value_name("DIR"),
        )
        .arg(
            Arg::with_name("debug")
                .help("Generate source level debug information (same as -C debuginfo=2)")
                .short("g")
                .long("debug"),
        )
        .arg(
            Arg::with_name("no-optimize")
                .help("Disable optimizations (optimization is enabled by default)")
                .long("no-optimize"),
        )
        .arg(
            Arg::with_name("opt-level")
                .conflicts_with("no-optimize")
                .long("opt-level")
                .short("O")
                .takes_value(true)
                .value_name("LEVEL")
                .default_value("2")
                .default_value_if("no-optimize", None, "0")
                .possible_values(&["0", "1", "2", "3", "s", "z"])
                .next_line_help(true)
                .help(
                    "\
                      Apply optimizations (default is -O2)\n  \
                        0 = no optimizations\n  \
                        1 = minimal optimizations\n  \
                        2 = normal optimizations (default)\n  \
                        3 = aggressive optimizations\n  \
                        s = optimize for size\n  \
                        z = aggressively optimize for size\n  \
                        _",
                ),
        )
        .arg(
            target
                .clone()
                .help("The target triple to compile against (e.g. x86_64-linux-gnu)"),
        )
        .arg(
            Arg::with_name("color")
                .help("Configure coloring of output")
                .next_line_help(true)
                .long("color")
                .possible_values(ColorArg::VARIANTS)
                .case_insensitive(true)
                .default_value("auto"),
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
            Arg::with_name("warnings-as-errors")
                .help("Causes the compiler to treat all warnings as errors")
                .long("warnings-as-errors"),
        )
        .arg(
            Arg::with_name("no-warn")
                .help("Disable warnings")
                .long("no-warn")
                .conflicts_with("warnings-as-errors"),
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
                     Example: `lumen compile -lc ...` will link against the system libc",
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
            Arg::with_name("append-path")
                .help("Appends a path to the Erlang code path")
                .long("append-path")
                .short("p")
                .value_name("PATH")
                .takes_value(true)
                .multiple(true)
                .number_of_values(1),
        )
        .arg(
            Arg::with_name("prepend-path")
                .help("Prepends a path to the Erlang code path")
                .long("prepend-path")
                .short("P")
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
