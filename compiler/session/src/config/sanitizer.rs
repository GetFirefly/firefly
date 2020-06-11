use std::fmt;
use std::str::FromStr;

use clap::ArgMatches;

use thiserror::Error;

use crate::config::options::{invalid_value, required_option_missing};
use crate::config::options::{OptionInfo, ParseOption};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum Sanitizer {
    Address,
    Leak,
    Memory,
    Thread,
}
impl fmt::Display for Sanitizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Sanitizer::Address => "address".fmt(f),
            Sanitizer::Leak => "leak".fmt(f),
            Sanitizer::Memory => "memory".fmt(f),
            Sanitizer::Thread => "thread".fmt(f),
        }
    }
}
impl FromStr for Sanitizer {
    type Err = InvalidSanitizerError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "address" => Ok(Sanitizer::Address),
            "leak" => Ok(Sanitizer::Leak),
            "memory" => Ok(Sanitizer::Memory),
            "thread" => Ok(Sanitizer::Thread),
            _ => Err(InvalidSanitizerError),
        }
    }
}
impl ParseOption for Sanitizer {
    fn parse_option<'a>(info: &OptionInfo, matches: &ArgMatches<'a>) -> clap::Result<Self> {
        match matches.value_of(info.name) {
            None => Err(required_option_missing(info)),
            Some(s) => Self::from_str(s).map_err(|e| invalid_value(info, &e.to_string())),
        }
    }
}

#[derive(Error, Debug, Clone, Copy, PartialEq)]
#[error("invalid sanitizer value")]
pub struct InvalidSanitizerError;
