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

Run with 'firefly -{} OPT[=VALUE]'

{{after-help}}
",
        option_group_help, option_group_prefix
    );

    let after_help = &format!(
        "You may only set one option at a time with '-{}'. \n\
         To set multiple options, you may pass '-{}' multiple times, \
         once for each option you need to set.\n\n\
         For more information, see the developer documentation at https:://github.com/lumen/firefly",
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

                for line in help.lines() {
                    // All lines we want to modify start with whitespace
                    if !line.starts_with(char::is_whitespace) {
                        println!("{}", line);
                        continue;
                    }
                    // Split on --, and if no split occurs, print the line
                    let parts = line.splitn(2, "--").collect::<Vec<_>>();
                    if parts.len() < 2 {
                        println!("{}", line);
                        continue;
                    }
                    // We want to do two things:
                    // 1. Strip the --, which we've accomplished by breaking up the line into parts
                    // 2. Reduce the amount of whitespace between the option name and help
                    print!("{}", &parts[0]);
                    let mut parts = parts[1].splitn(2, "  ");
                    let option = parts.next().unwrap();
                    print!("{}", option);
                    if let Some(remainder) = parts.next() {
                        println!("{}", &remainder[20..])
                    } else {
                        println!()
                    }
                }
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
