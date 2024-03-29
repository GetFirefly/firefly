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
impl Default for DebugInfo {
    fn default() -> Self {
        Self::Full
    }
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
        matches
            .value_of(info.name)
            .map_or(Ok(Self::Full), |s| s.parse())
            .map_err(|_| invalid_value(info, &format!("expected debug level 0-2")))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Hash)]
pub enum SplitDwarfKind {
    /// Sections which do not require relocation are written into object file but ignored by the
    /// linker.
    Single,
    /// Sections which do not require relocation are written into a DWARF object (`.dwo`) file
    /// which is ignored by the linker.
    Split,
}
impl Default for SplitDwarfKind {
    fn default() -> Self {
        Self::Split
    }
}
impl FromStr for SplitDwarfKind {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, ()> {
        Ok(match s {
            "single" => Self::Single,
            "split" => Self::Split,
            _ => return Err(()),
        })
    }
}
impl ParseOption for SplitDwarfKind {
    fn parse_option<'a>(info: &OptionInfo, matches: &ArgMatches<'a>) -> clap::Result<Self> {
        matches
            .value_of(info.name)
            .map_or(Ok(Self::Split), |s| s.parse())
            .map_err(|_| invalid_value(info, &format!("expected valid split dwarf kind")))
    }
}

/// The different settings that the `-Z strip` flag can have.
#[derive(Clone, Copy, PartialEq, Hash, Debug)]
pub enum Strip {
    /// Do not strip at all.
    None,
    /// Strip debuginfo.
    DebugInfo,
    /// Strip all symbols.
    Symbols,
}
impl Default for Strip {
    fn default() -> Self {
        Self::None
    }
}
impl FromStr for Strip {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "none" => Ok(Self::None),
            "debuginfo" => Ok(Self::DebugInfo),
            "symbols" => Ok(Self::Symbols),
            _ => Err(()),
        }
    }
}
impl ParseOption for Strip {
    fn parse_option<'a>(info: &OptionInfo, matches: &ArgMatches<'a>) -> clap::Result<Self> {
        matches.value_of(info.name).map_or_else(
            || Err(required_option_missing(info)),
            |s| Self::from_str(s).map_err(|_| invalid_value(info, "invalid strip setting")),
        )
    }
}
