use std::error::Error;
use std::path::PathBuf;
use std::str::FromStr;

use clap::{ArgMatches, ErrorKind};

use liblumen_target as target;
use liblumen_target::spec::{
    LinkerFlavor, MergeFunctions, PanicStrategy, RelroLevel, Target, TargetError,
};
use liblumen_util::diagnostics::ColorArg;

use super::OptionInfo;

pub trait ParseOption: Sized {
    fn parse_option<'a>(info: &OptionInfo, matches: &ArgMatches<'a>) -> clap::Result<Self>;
}
impl<T> ParseOption for Option<T>
where
    T: ParseOption,
{
    fn parse_option<'a>(info: &OptionInfo, matches: &ArgMatches<'a>) -> clap::Result<Self> {
        if matches.is_present(info.name) {
            match T::parse_option(info, matches) {
                Ok(val) => Ok(Some(val)),
                Err(err) => Err(err),
            }
        } else {
            Ok(None)
        }
    }
}
impl<T, E> ParseOption for Vec<T>
where
    E: Error,
    T: FromStr<Err = E>,
{
    fn parse_option<'a>(info: &OptionInfo, matches: &ArgMatches<'a>) -> clap::Result<Self> {
        match matches.values_of(info.name) {
            None => Ok(Vec::new()),
            Some(values) => {
                let mut result = Vec::new();
                for value in values {
                    match T::from_str(value) {
                        Err(err) => {
                            return Err(clap::Error {
                                kind: ErrorKind::ValueValidation,
                                message: err.to_string(),
                                info: Some(vec![info.name.to_string()]),
                            });
                        }
                        Ok(value) => result.push(value),
                    }
                }
                Ok(result)
            }
        }
    }
}

impl ParseOption for String {
    fn parse_option<'a>(info: &OptionInfo, matches: &ArgMatches<'a>) -> clap::Result<Self> {
        match matches.value_of(info.name) {
            None => Err(required_option_missing(info)),
            Some(s) => Ok(s.to_string()),
        }
    }
}
impl ParseOption for bool {
    fn parse_option<'a>(info: &OptionInfo, matches: &ArgMatches<'a>) -> clap::Result<Self> {
        Ok(matches.is_present(info.name))
    }
}
impl ParseOption for u64 {
    fn parse_option<'a>(info: &OptionInfo, matches: &ArgMatches<'a>) -> clap::Result<Self> {
        match matches.value_of(info.name) {
            None => Err(required_option_missing(info)),
            Some(s) => u64::from_str(s).map_err(|e| invalid_value(info, &e.to_string())),
        }
    }
}
impl ParseOption for PathBuf {
    fn parse_option<'a>(info: &OptionInfo, matches: &ArgMatches<'a>) -> clap::Result<Self> {
        Ok(matches
            .value_of(info.name)
            .map(PathBuf::from)
            .unwrap_or_else(Default::default))
    }
}
impl ParseOption for LinkerFlavor {
    fn parse_option<'a>(info: &OptionInfo, matches: &ArgMatches<'a>) -> clap::Result<Self> {
        match matches.value_of(info.name) {
            None => Err(required_option_missing(info)),
            Some(s) => LinkerFlavor::from_str(s).map_err(|e| invalid_value(info, &e.to_string())),
        }
    }
}
impl ParseOption for RelroLevel {
    fn parse_option<'a>(info: &OptionInfo, matches: &ArgMatches<'a>) -> clap::Result<Self> {
        match matches.value_of(info.name) {
            None => Err(required_option_missing(info)),
            Some(s) => {
                RelroLevel::from_str(s).map_err(|_| invalid_value(info, "invalid relro level type"))
            }
        }
    }
}
impl ParseOption for MergeFunctions {
    fn parse_option<'a>(info: &OptionInfo, matches: &ArgMatches<'a>) -> clap::Result<Self> {
        match matches.value_of(info.name) {
            None => Err(required_option_missing(info)),
            Some(s) => MergeFunctions::from_str(s)
                .map_err(|_| invalid_value(info, "invalid merge functions type")),
        }
    }
}
impl ParseOption for PanicStrategy {
    fn parse_option<'a>(info: &OptionInfo, matches: &ArgMatches<'a>) -> clap::Result<Self> {
        matches
            .value_of(info.name)
            .map_or(Ok(Self::Unwind), |s| s.parse())
            .map_err(|_| {
                invalid_value(
                    info,
                    "unrecognized panic strategy, expected 'unwind' or 'abort'",
                )
            })
    }
}
impl ParseOption for Target {
    fn parse_option<'a>(info: &OptionInfo, matches: &ArgMatches<'a>) -> clap::Result<Self> {
        let triple = match matches.value_of(info.name) {
            None => target::host_triple(),
            Some(s) => s,
        };
        match Target::search(triple) {
            Ok(target) => Ok(target),
            Err(err @ TargetError::Unsupported(_)) => Err(clap::Error {
                kind: ErrorKind::ValueValidation,
                message: err.to_string(),
                info: Some(vec![info.name.to_string()]),
            }),
            Err(TargetError::Other(desc)) => Err(clap::Error {
                kind: ErrorKind::ValueValidation,
                message: format!("unable to load target: {}", desc),
                info: Some(vec![info.name.to_string()]),
            }),
        }
    }
}
impl ParseOption for ColorArg {
    fn parse_option<'a>(info: &OptionInfo, matches: &ArgMatches<'a>) -> clap::Result<Self> {
        let choice = match matches.value_of(info.name) {
            None => "auto",
            Some(s) => s,
        };
        choice.parse().map_err(|e| invalid_value(info, e))
    }
}

pub(in crate::config) fn invalid_value(info: &OptionInfo, description: &str) -> clap::Error {
    clap::Error {
        kind: ErrorKind::InvalidValue,
        message: description.to_string(),
        info: Some(vec![info.name.to_string()]),
    }
}

pub(in crate::config) fn required_option_missing(info: &OptionInfo) -> clap::Error {
    clap::Error {
        kind: ErrorKind::MissingRequiredArgument,
        message: format!("required argument was not provided"),
        info: Some(vec![info.name.to_string()]),
    }
}
