use std::fmt;
use std::str::FromStr;

use clap::ArgMatches;

use crate::config::options::{invalid_value, required_option_missing};
use crate::config::options::{OptionInfo, ParseOption};

/// The type of project we're building
#[derive(Copy, PartialEq, PartialOrd, Clone, Ord, Eq, Hash, Debug)]
pub enum ProjectType {
    Executable,
    Dylib,
    Staticlib,
    Cdylib,
}
impl ProjectType {
    pub fn is_executable(&self) -> bool {
        if let Self::Executable = self {
            return true;
        }
        false
    }
}
impl Default for ProjectType {
    fn default() -> Self {
        ProjectType::Executable
    }
}
impl fmt::Display for ProjectType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            ProjectType::Executable => "bin".fmt(f),
            ProjectType::Dylib => "dylib".fmt(f),
            ProjectType::Staticlib => "staticlib".fmt(f),
            ProjectType::Cdylib => "cdylib".fmt(f),
        }
    }
}
impl FromStr for ProjectType {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "lib" | "dylib" => Ok(Self::Dylib),
            "staticlib" => Ok(Self::Staticlib),
            "cdylib" => Ok(Self::Cdylib),
            "bin" => Ok(Self::Executable),
            _ => Err(()),
        }
    }
}
impl ParseOption for ProjectType {
    fn parse_option<'a>(info: &OptionInfo, matches: &ArgMatches<'a>) -> clap::Result<Self> {
        match matches.value_of(info.name) {
            None => Err(required_option_missing(info)),
            Some(s) => s
                .parse()
                .map_err(|_| invalid_value(info, &format!("unknown project type: `{}`", s))),
        }
    }
}
