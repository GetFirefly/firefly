mod parser;

use proc_macro::TokenStream;

use quote::quote;

pub use self::parser::{OptionGroupConfig, OptionGroupStruct, OptionInfo};

/// Handles generating the configured option group struct definition,
/// associated functions and trait implementations.
pub fn generate_option_group(
    config: OptionGroupConfig,
    option_group: OptionGroupStruct,
) -> TokenStream {
    let option_group_name = config.name.as_str();
    let option_group_help = config.help.as_str();
    let option_group_prefix = config.short.as_str();

    let struct_name = option_group.def.ident.clone();
    let field_name = &option_group
        .def
        .fields
        .iter()
        .map(|f| f.ident.clone().unwrap())
        .collect::<Vec<_>>();
    let option = &option_group
        .options
        .iter()
        .map(|o| o.to_option_info_args())
        .collect::<Vec<_>>();
    let arg = &option_group
        .options
        .iter()
        .map(|o| o.to_arg())
        .collect::<Vec<_>>();

    let template = &format!(
        "
{}:

{{unified}}

Run with 'lumen -{} OPT[=VALUE]'

{{after-help}}
",
        option_group_help, option_group_prefix
    );

    let after_help = &format!(
        "You may only set one option at a time with '-{}'. \n\
         To set multiple options, you may pass '-{}' multiple times, \
         once for each option you need to set.\n\n\
         For more information, see the developer documentation at https:://github.com/lumen/lumen",
        option_group_prefix, option_group_prefix
    );
    let expanded = quote! {
        #option_group
        impl #struct_name {
            const OPTIONS: &'static [crate::config::OptionInfo] = &[
                #(
                    crate::config::OptionInfo::new(#option),
                )*
            ];
        }
        impl crate::config::OptionGroup for #struct_name {
            fn option_group_name() -> &'static str {
                #option_group_name
            }

            fn option_group_help() -> &'static str {
                #option_group_help
            }

            fn option_group_prefix() -> &'static str {
                #option_group_prefix
            }

            fn option_group_arg<'a, 'b>() -> clap::Arg<'a, 'b> {
                use clap::Arg;

                Arg::with_name(#option_group_name)
                    .help(#option_group_help)
                    .short(#option_group_prefix)
                    .takes_value(true)
                    .value_name("OPT[=VALUE]")
                    .multiple(true)
                    .number_of_values(1)
            }

            fn option_group_app<'a, 'b>() -> clap::App<'a, 'b> {
                use clap::{App, AppSettings, Arg};
                App::new(#option_group_name)
                    .setting(AppSettings::ArgsNegateSubcommands)
                    .setting(AppSettings::DisableHelpFlags)
                    .setting(AppSettings::DisableHelpSubcommand)
                    .setting(AppSettings::DisableVersion)
                    .setting(AppSettings::NoBinaryName)
                    .setting(AppSettings::UnifiedHelpMessage)
                    .template(#template)
                    .after_help(#after_help)
                    .subcommand(App::new("help").about("Print all options in this option group"))
                    .args(&[
                        #(#arg,)*
                    ])
            }

            fn print_help() {
                use clap::{App, AppSettings, Arg};

                let mut app = Self::option_group_app();
                let mut buf: Vec<u8> = Vec::new();
                app.write_help(&mut buf).expect("unable to print help");
                let help = std::str::from_utf8(buf.as_slice())
                    .expect("unable to print help due to invalid encoding, this is a bug!");

                help.lines().take(3).for_each(|line| {
                    println!("{}", line)
                });
                help.lines()
                    .skip(3)
                    .for_each(|line| {
                        let mut split = line.splitn(2, "--");
                        let leading_space = split.next().unwrap();
                        if let Some(arg) = split.next() {
                            println!("    {}", arg);
                        } else {
                            if leading_space.starts_with(|c: char| c.is_whitespace()) {
                                println!("{}", leading_space.chars().skip(6).collect::<String>());
                            } else {
                                println!("{}", leading_space);
                            }
                        }
                    });
            }

            fn option_group_options() -> &'static [crate::config::options::OptionInfo] {
                Self::OPTIONS
            }

            fn parse_option_group<'a>(matches: &clap::ArgMatches<'a>) -> crate::config::options::OptionGroupParseResult<Self> {
                use crate::config::options::{OptionGroupParser, OptionGroupParseResult};
                match matches.values_of(#option_group_name) {
                    None => {
                        let parser = OptionGroupParser::default();
                        parser.parse()
                    }
                    Some(values) => {
                        let parser = OptionGroupParser::new(values);
                        parser.parse()
                    }
                }
            }
        }
        impl<'a> std::convert::TryFrom<clap::ArgMatches<'a>> for #struct_name {
            type Error = clap::Error;

            fn try_from(matches: clap::ArgMatches<'a>) -> Result<Self, Self::Error> {
                use std::collections::HashMap;
                use crate::config::{ParseOption, OptionInfo};

                let mut base = Self::default();
                let info: HashMap<&'static str, &OptionInfo> = Self::OPTIONS.iter()
                    .map(|o| (o.name, o))
                    .collect();
                #(
                    base.#field_name = ParseOption::parse_option(
                        info.get(stringify!(#field_name)).unwrap(),
                        &matches,
                    )?;
                )*
                Ok(base)
            }
        }
    };

    TokenStream::from(expanded)
}
