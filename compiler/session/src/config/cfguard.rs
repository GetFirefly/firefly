use std::str::FromStr;

use clap::ArgMatches;

use crate::config::options::{invalid_value, required_option_missing};
use crate::config::options::{OptionInfo, ParseOption};

/// The different settings that the `-C control-flow-guard` flag can have.
#[derive(Clone, Copy, PartialEq, Hash, Debug)]
pub enum CFGuard {
    /// Do not emit Control Flow Guard metadata or checks.
    Disabled,

    /// Emit Control Flow Guard metadata but no checks.
    NoChecks,

    /// Emit Control Flow Guard metadata and checks.
    Checks,
}
impl Default for CFGuard {
    fn default() -> Self {
        Self::Disabled
    }
}

impl FromStr for CFGuard {
    type Err = ();

    fn from_str(s: &str) -> Result<CFGuard, ()> {
        match s {
            "disabled" | "false" => Ok(CFGuard::Disabled),
            "nochecks" => Ok(CFGuard::NoChecks),
            "true" | "checks" => Ok(CFGuard::Checks),
            _ => Err(()),
        }
    }
}

impl ParseOption for CFGuard {
    fn parse_option<'a>(info: &OptionInfo, matches: &ArgMatches<'a>) -> clap::Result<Self> {
        matches.value_of(info.name).map_or_else(
            || Err(required_option_missing(info)),
            |s| {
                Self::from_str(s)
                    .map_err(|_| invalid_value(info, "invalid control flow guard option"))
            },
        )
    }
}
