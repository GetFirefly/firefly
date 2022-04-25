use std::path::PathBuf;
use std::str::FromStr;

use clap::ArgMatches;

use crate::config::options::{invalid_value, required_option_missing};
use crate::config::{OptionInfo, ParseOption};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LdImpl {
    Lld,
}
impl FromStr for LdImpl {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "lld" => Ok(Self::Lld),
            _ => Err(()),
        }
    }
}
impl ParseOption for LdImpl {
    fn parse_option<'a>(info: &OptionInfo, matches: &ArgMatches<'a>) -> clap::Result<Self> {
        matches.value_of(info.name).map_or_else(
            || Err(required_option_missing(info)),
            |s| {
                Self::from_str(s)
                    .map_err(|_| invalid_value(info, "expected valid gcc-ld implementation"))
            },
        )
    }
}

/// The linker plugin to use
#[derive(Debug, Clone, PartialEq, Hash)]
pub enum LinkerPluginLto {
    Plugin(PathBuf),
    Auto,
    Disabled,
}
impl Default for LinkerPluginLto {
    fn default() -> Self {
        Self::Auto
    }
}
impl LinkerPluginLto {
    pub fn enabled(&self) -> bool {
        match *self {
            Self::Plugin(_) | Self::Auto => true,
            Self::Disabled => false,
        }
    }
}
impl From<&str> for LinkerPluginLto {
    fn from(s: &str) -> Self {
        match s {
            "" => Self::default(),
            "false" | "disabled" => Self::Disabled,
            path => Self::Plugin(PathBuf::from(path)),
        }
    }
}
impl ParseOption for LinkerPluginLto {
    fn parse_option<'a>(info: &OptionInfo, matches: &ArgMatches<'a>) -> clap::Result<Self> {
        Ok(matches
            .value_of(info.name)
            .map(Self::from)
            .unwrap_or(Self::Auto))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WasiExecModel {
    Command,
    Reactor,
}
impl FromStr for WasiExecModel {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "command" => Ok(Self::Command),
            "reactor" => Ok(Self::Reactor),
            _ => Err(()),
        }
    }
}
impl ParseOption for WasiExecModel {
    fn parse_option<'a>(info: &OptionInfo, matches: &ArgMatches<'a>) -> clap::Result<Self> {
        matches.value_of(info.name).map_or_else(
            || Err(required_option_missing(info)),
            |s| Self::from_str(s).map_err(|_| invalid_value(info, "invalid wasi exec model")),
        )
    }
}
