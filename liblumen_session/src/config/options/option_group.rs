use std::convert::TryFrom;
use std::marker::PhantomData;

use thiserror::Error;

use clap::{ArgMatches, ErrorKind, Values};

use super::OptionInfo;

#[derive(Error, Debug, Clone, PartialEq, Eq)]
#[error("show help for option group '-{0}'")]
pub struct ShowOptionGroupHelp(&'static str);
impl ShowOptionGroupHelp {
    pub fn prefix(&self) -> &'static str {
        self.0
    }

    pub fn print_help(&self) {
        use crate::config::{CodegenOptions, DebuggingOptions};

        if self.0 == CodegenOptions::option_group_prefix() {
            CodegenOptions::print_help();
        } else if self.0 == DebuggingOptions::option_group_prefix() {
            DebuggingOptions::print_help();
        } else {
            panic!("unexpected option group prefix '{}', this is a bug");
        }
    }
}

pub trait OptionGroup: Sized {
    /// Returns the name of this option group
    fn option_group_name() -> &'static str;
    /// Returns the help documentation for this option group
    fn option_group_help() -> &'static str;
    /// Returns the short flag prefix for this option group, i.e. `C`,
    /// which gets used as `-C OPT[=VAL]` on the command line.
    fn option_group_prefix() -> &'static str;

    /// Returns the `clap::Arg` which instantiates this option group
    /// as an argument in a higher-level `clap::App`.
    ///
    /// The argument is implicitly a multiple occurrance, single value
    /// argument, handling values of the form `OPT[=VALUE]`, where `VALUE`
    /// is optional and may contain any type of value, including values with
    /// leading hyphens.
    ///
    /// Parsing of the argument values is handled by the associated `OptionGroupParser`
    fn option_group_arg<'a, 'b>() -> clap::Arg<'a, 'b>;

    /// Returns the `clap::App` which defines arguments for this option group
    fn option_group_app<'a, 'b>() -> clap::App<'a, 'b>;

    /// Prints the help/usage for this option group to stdout
    fn print_help();

    /// Return the set of `OptionInfo`s that define options belonging to this group
    fn option_group_options() -> &'static [OptionInfo];

    /// Parses this option group from the given `ArgMatches`, by looking up the prefix as an
    /// argument, then parsing the values given to that argument.
    ///
    /// This is the primary way you should parse an option group
    fn parse_option_group<'a>(matches: &ArgMatches<'a>) -> super::OptionGroupParseResult<Self>;
}

pub struct OptionGroupParser<T: OptionGroup> {
    args: Vec<String>,
    _marker: PhantomData<T>,
}
impl<T> OptionGroupParser<T>
where
    T: OptionGroup,
{
    pub fn new<'a>(values: Values<'a>) -> Self {
        let mut args = Vec::new();
        for value in values {
            let mut split: Vec<_> = value.splitn(2, '=').collect();
            let value = split.pop().unwrap();
            if let Some(key) = split.pop() {
                args.push(format!("--{}={}", key, value));
            } else {
                // pop() removes from the back
                args.push(format!("--{}", value));
            }
        }
        Self {
            args,
            _marker: PhantomData,
        }
    }
}
impl<'a, T> OptionGroupParser<T>
where
    T: OptionGroup + TryFrom<clap::ArgMatches<'a>, Error = clap::Error>,
{
    pub fn parse(self) -> OptionGroupParseResult<T> {
        let app = T::option_group_app();
        let matches = Self::coerce_help(app.get_matches_from_safe(self.args))?;
        println!("matches: {:#?}", &matches);
        match matches.subcommand_name() {
            Some(unknown) => Err(clap::Error {
                kind: ErrorKind::UnrecognizedSubcommand,
                message: format!(
                    "expected `help`, or an option, but got `{}` which is unrecognized",
                    unknown
                ),
                info: None,
            }
            .into()),
            None => match T::try_from(matches) {
                Ok(value) => Ok(Some(value)),
                Err(err) => Err(err.into()),
            },
        }
    }

    fn coerce_help<'b>(
        maybe_matches: clap::Result<ArgMatches<'b>>,
    ) -> Result<ArgMatches<'b>, anyhow::Error> {
        if let Ok(matches) = maybe_matches {
            return Ok(matches);
        }
        if let Err(clap::Error {
            kind: ErrorKind::UnknownArgument,
            info: Some(info),
            message,
        }) = maybe_matches
        {
            match info[0].as_str() {
                "--help" | "help" => {
                    return Err(ShowOptionGroupHelp(T::option_group_prefix()).into())
                }
                _ => {
                    return Err(clap::Error {
                        kind: ErrorKind::UnknownArgument,
                        info: Some(info),
                        message,
                    }
                    .into())
                }
            }
        } else {
            maybe_matches.map_err(|e| e.into())
        }
    }
}

pub type OptionGroupParseResult<T> = Result<Option<T>, anyhow::Error>;
