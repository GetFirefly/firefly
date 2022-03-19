use std::str::FromStr;

use clap::ArgMatches;

use crate::config::options::invalid_value;
use crate::config::options::{OptionInfo, ParseOption};

/// Represents how MLIR debug info is printed
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum MlirDebugPrinting {
    /// Disables debug info printing
    None,
    /// Enables printing of debug info in a form that is parseable by MLIR
    Plain,
    /// Enables printing of debug info in an easy-to-read form, albeit unparseable by MLIR
    Pretty,
}
impl Default for MlirDebugPrinting {
    fn default() -> Self {
        Self::None
    }
}

impl FromStr for MlirDebugPrinting {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "false" | "none" => Ok(Self::None),
            "plain" => Ok(Self::Plain),
            "pretty" => Ok(Self::Pretty),
            _ => Err(()),
        }
    }
}

impl ParseOption for MlirDebugPrinting {
    fn parse_option<'a>(info: &OptionInfo, matches: &ArgMatches<'a>) -> clap::Result<Self> {
        matches
            .value_of(info.name)
            .map_or(Ok(Self::None), |s| s.parse())
            .map_err(|_| {
                invalid_value(
                    info,
                    "unrecognized debug printing option, expected 'none', 'plain' or 'pretty'",
                )
            })
    }
}
