use std::path::PathBuf;

use anyhow::{anyhow, bail};
use clap::Args;
use lit::event_handler::Default as EventHandler;

#[derive(Args)]
pub struct Config {
    /// Path to the Firefly workspace.
    ///
    /// If not specified, uses the cargo-make workspace directory or the current working directory
    #[clap(long, env("CARGO_MAKE_WORKSPACE_WORKING_DIRECTORY"), value_parser)]
    workspace: Option<PathBuf>,

    /// Path to the directory containing the lit tests to be run
    #[clap(value_parser)]
    tests: PathBuf,

    /// Specify an extension to consider when looking for files to parse for tests
    #[clap(long = "extension", short = 't', default_value = "erl", action = clap::ArgAction::Append, value_parser = clap::builder::NonEmptyStringValueParser::new())]
    file_types: Vec<String>,

    #[clap(last(true))]
    only: Vec<String>,
}

pub fn run(config: &Config) -> anyhow::Result<()> {
    let workspace = config
        .workspace
        .as_ref()
        .cloned()
        .unwrap_or_else(|| std::env::current_dir().unwrap());

    let test_path = config.tests.as_path();
    let lit_dir = if test_path.is_file() {
        test_path.parent().unwrap().to_str().unwrap()
    } else {
        test_path.to_str().unwrap()
    };

    let firefly_exe = workspace.join("bin/firefly");
    if !firefly_exe.is_file() {
        bail!(
            "expected to find compiler at {}, but it either doesn't exist or is not a file",
            firefly_exe.display()
        );
    }

    lit::run::tests(EventHandler::default(), |runner| {
        runner.add_search_path(test_path.to_str().unwrap());
        for ty in config.file_types.iter() {
            runner.add_extension(ty.as_str());
        }

        runner
            .constants
            .insert("tests".to_owned(), lit_dir.to_string());
        runner.constants.insert(
            "firefly".to_owned(),
            firefly_exe.to_str().unwrap().to_string(),
        );
    })
    .map_err(|_| anyhow!("lit tests failed, see output for details"))
}
