use std::str::FromStr;

use clap::ArgMatches;

use crate::config::options::{invalid_value, required_option_missing};
use crate::config::options::{OptionInfo, ParseOption};

/// The level of debug info to generate
#[derive(Debug, Clone, Copy, PartialEq, Hash)]
pub enum DebugInfo {
    None,
    Limited,
    Full,
}
impl FromStr for DebugInfo {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "0" => Ok(Self::None),
            "1" => Ok(Self::Limited),
            "2" => Ok(Self::Full),
            _ => Err(()),
        }
    }
}
impl ParseOption for DebugInfo {
    fn parse_option<'a>(info: &OptionInfo, matches: &ArgMatches<'a>) -> clap::Result<Self> {
        matches.value_of(info.name).map_or_else(
            || Err(required_option_missing(info)),
            |s| Self::from_str(s).map_err(|_| invalid_value(info, "invalid debug info level")),
        )
    }
}
