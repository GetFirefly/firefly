use std::collections::HashMap;
use std::error::Error;
use std::ffi::{OsStr, OsString};
use std::fs;
use std::io;
use std::path::Path;

use clap::{App, AppSettings, Arg, SubCommand};

pub type ConfigResult<T> = std::result::Result<T, ConfigError>;
//TODO: Needs to be HashMap<Atom, HashMap<Atom, Term>>
pub type AppConfig = HashMap<String, HashMap<String, String>>;
pub type BootScript = Vec<BootInstruction>;
//TODO: Needs to be Term
pub type BootInstruction = String;

#[derive(Debug)]
pub enum Command {
    Run,
    Shell,
    RemoteShell(String),
}

#[derive(Debug)]
pub enum ConfigError {
    FileError(OsString, io::Error),
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
            ConfigError::FileError(ref path, ref err) => write!(
                f,
                "Failed to load {}: {}",
                path.to_string_lossy(),
                err.description()
            ),
        }
    }
}

impl std::error::Error for ConfigError {
    fn description(&self) -> &str {
        match *self {
            ConfigError::FileError(_, ref err) => err.description(),
        }
    }
    fn cause(&self) -> Option<&std::error::Error> {
        match *self {
            ConfigError::FileError(ref _path, ref err) => Some(err),
        }
    }
}

#[derive(Debug)]
pub struct Config {
    pub config: AppConfig,
    pub boot: Option<BootScript>,
    pub debug: bool,
    pub name: Option<String>,
    pub cookie: Option<String>,
    pub command: Command,
    pub extra: Vec<String>,
}

impl Config {
    pub fn from_argv(app: String, version: String, argv: Vec<String>) -> ConfigResult<Config> {
        let matches = App::new(app)
            .version(version.as_str())
            .setting(AppSettings::TrailingVarArg)
            .arg(Arg::with_name("args_file")
                     .long("args_file")
                     .help("Provide a path to a vm.args file containing VM configuration")
                     .takes_value(true)
                     .multiple(true)
                     .number_of_values(1))
            .arg(Arg::with_name("config")
                     .long("config")
                     .help("Provide a path to a sys.config file containing application configuration")
                     .takes_value(true)
                     .multiple(true)
                     .number_of_values(1))
            .arg(Arg::with_name("boot")
                     .long("boot")
                     .help("Provide a path to a .boot or .script file which defines how to boot the system")
                     .takes_value(true))
            .arg(Arg::with_name("debug")
                     .long("debug")
                     .help("Enable debug output from the runtime"))
            .arg(Arg::with_name("name")
                     .long("name")
                     .global(true)
                     .help("The secret cookie to use in distributed mode\n\
                            If one is not provided, one will be generated for you in ~/.erlang.cookie")
                     .takes_value(true)
                     .validator(is_valid_node_name))
            .arg(Arg::with_name("cookie")
                     .long("cookie")
                     .global(true)
                     .help("The secret cookie to use in distributed mode")
                     .takes_value(true)
                     .env("COOKIE"))
            .arg(Arg::with_name("extra")
                     .last(true)
                     .multiple(true)
                     .hidden(true))
            .subcommand(
                SubCommand::with_name("shell")
                    .about("Starts a new interactive shell, but does not start the system")
                    .arg(Arg::with_name("remote")
                            .long("remote")
                            .help("Connects a remote shell to the specified host")
                            .takes_value(true)
                            .validator(is_valid_node_name)))
            .get_matches_from(argv);

        let command: Command;
        let extra: Vec<&str>;
        if let Some(matches) = matches.subcommand_matches("shell") {
            if let Some(target) = matches.value_of("remote") {
                command = Command::RemoteShell(target.to_string());
            } else {
                command = Command::Shell;
            }
            extra = Vec::new();
        } else {
            extra = match matches.values_of("extra") {
                None => Vec::new(),
                Some(vs) => vs.collect(),
            };
            command = Command::Run;
        }
        Ok(Config {
            config: with_file(
                matches.value_of_os("config"),
                AppConfig::new(),
                load_app_config,
            )?,
            boot: with_file(matches.value_of_os("boot"), None, load_boot_script)?,
            debug: matches.is_present("debug"),
            name: matches.value_of("name").map(|v| v.to_string()),
            cookie: matches.value_of("cookie").map(|v| v.to_string()),
            command,
            extra: extra.iter().map(|v| v.to_string()).collect(),
        })
    }
}

fn is_valid_node_name(_f: String) -> Result<(), String> {
    //TODO: Validate name
    Ok(())
}

fn with_file<T>(v: Option<&OsStr>, default: T, fun: fn(String) -> T) -> ConfigResult<T> {
    match v {
        None => Ok(default),
        Some(p) => {
            let path = Path::new(p);
            match fs::read_to_string(path) {
                Err(err) => Err(ConfigError::FileError(p.to_os_string(), err)),
                Ok(contents) => Ok(fun(contents)),
            }
        }
    }
}

fn load_app_config(_contents: String) -> AppConfig {
    AppConfig::new()
}

fn load_boot_script(_contents: String) -> Option<BootScript> {
    None
}
