use std::convert::From;
use std::path::PathBuf;
use std::str::FromStr;

use clap::ArgMatches;

use crate::config::options::invalid_value;
use crate::config::options::{OptionInfo, ParseOption};

/// The optimization level to use during compilation
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum OptLevel {
    No,         // -O0
    Less,       // -O1
    Default,    // -O2
    Aggressive, // -O3
    Size,       // -Os
    SizeMin,    // -Oz
}
impl Default for OptLevel {
    fn default() -> Self {
        Self::No
    }
}
impl FromStr for OptLevel {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, ()> {
        match s {
            "0" => Ok(Self::No),
            "1" => Ok(Self::Less),
            "2" => Ok(Self::Default),
            "3" => Ok(Self::Aggressive),
            "s" => Ok(Self::Size),
            "z" => Ok(Self::SizeMin),
            _ => Err(()),
        }
    }
}
impl ParseOption for OptLevel {
    fn parse_option<'a>(info: &OptionInfo, matches: &ArgMatches<'a>) -> clap::Result<Self> {
        matches
            .value_of(info.name)
            .map_or(Ok(OptLevel::No), |s| s.parse())
            .map_err(|_| invalid_value(info, &format!("expected optimization level 0-3, s, or z")))
    }
}

/// This is what the `LtoCli` values get mapped to after resolving defaults and
/// and taking other command line options into account.
#[derive(Debug, Clone, PartialEq)]
pub enum Lto {
    /// Don't do any LTO whatsoever
    No,
    /// Do a full crate graph LTO with ThinLTO
    Thin,
    /// Do a local graph LTO with ThinLTO (only relevant for multiple codegen
    /// units).
    ThinLocal,
    /// Do a full crate graph LTO with "fat" LTO
    Fat,
}

/// The different settings that the `-C lto` flag can have.
#[derive(Clone, Copy, PartialEq, Hash, Debug)]
pub enum LtoCli {
    /// `-C lto=no`
    No,
    /// `-C lto=yes`
    Yes,
    /// `-C lto=thin`
    Thin,
    /// `-C lto=fat`
    Fat,
    /// No `-C lto` flag passed
    Unspecified,
}
impl Default for LtoCli {
    fn default() -> Self {
        Self::Unspecified
    }
}
impl FromStr for LtoCli {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "" => Ok(Self::default()),
            "thin" => Ok(Self::Thin),
            "fat" => Ok(Self::Fat),
            "yes" | "true" => Ok(Self::Yes),
            "no" | "false" => Ok(Self::No),
            _ => Err(()),
        }
    }
}
impl ParseOption for LtoCli {
    fn parse_option<'a>(info: &OptionInfo, matches: &ArgMatches<'a>) -> clap::Result<Self> {
        matches
            .value_of(info.name)
            .map_or(Ok(LtoCli::Unspecified), |s| s.parse())
            .map_err(|_| {
                invalid_value(
                    info,
                    "unrecognized lto type, expected one of thin|fat|true|false|yes|no",
                )
            })
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
            LinkerPluginLto::Plugin(_) | LinkerPluginLto::Auto => true,
            LinkerPluginLto::Disabled => false,
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

#[derive(Debug, Clone, Hash)]
pub enum Passes {
    Some(Vec<String>),
    All,
}
impl Default for Passes {
    fn default() -> Self {
        Self::Some(Vec::new())
    }
}
impl ParseOption for Passes {
    fn parse_option<'a>(info: &OptionInfo, matches: &ArgMatches<'a>) -> clap::Result<Self> {
        match matches.values_of(info.name) {
            None => Ok(Self::default()),
            Some(ps) => {
                let ps = ps.collect::<Vec<_>>();
                if ps.len() == 1 && ps[0] == "all" {
                    return Ok(Passes::All);
                }
                let mut passes = Vec::new();
                for pass in ps {
                    passes.push(pass.to_string());
                }
                Ok(Passes::Some(passes))
            }
        }
    }
}
impl Passes {
    pub fn is_empty(&self) -> bool {
        match *self {
            Passes::Some(ref v) => v.is_empty(),
            Passes::All => false,
        }
    }
}
