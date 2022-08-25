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

    /// The extension of the files that should be parsed for tests
    #[clap(long, default_value = "erl", value_parser = clap::builder::NonEmptyStringValueParser::new())]
    file_type: String,
}

pub fn run(config: &Config) -> anyhow::Result<()> {
    let workspace = config
        .workspace
        .as_ref()
        .cloned()
        .unwrap_or_else(|| std::env::current_dir().unwrap());
    let lit_dir = config.tests.as_path().to_str().unwrap();

    let lumen_exe = workspace.join("bin/lumen");
    if !lumen_exe.is_file() {
        bail!(
            "expected to find compiler at {}, but it either doesn't exist or is not a file",
            lumen_exe.display()
        );
    }

    lit::run::tests(EventHandler::default(), |runner| {
        runner.add_search_path(lit_dir);
        runner.add_extension(config.file_type.as_str());

        runner
            .constants
            .insert("tests".to_owned(), lit_dir.to_string());
        runner
            .constants
            .insert("lumen".to_owned(), lumen_exe.to_str().unwrap().to_string());
    })
    .map_err(|_| anyhow!("lit tests failed, see output for details"))
}
