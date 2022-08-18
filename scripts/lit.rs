//! ```cargo
//! [dependencies]
//! lit = { git = "https://github.com/lumen/lit", branch = "erlang" }
//! ```

extern crate lit;

use std::env;
use std::path::PathBuf;

const WORKSPACE_DIR: &'static str = env!("CARGO_MAKE_WORKING_DIRECTORY");

fn main() {
    let workspace_dir = PathBuf::from(WORKSPACE_DIR);
    let lit_dir = workspace_dir.join("test/lit");
    let lumen_exe = workspace_dir.join("bin/lumen");

    lit::run::tests(lit::event_handler::Default::default(), |config| {
        config.add_search_path(lit_dir.to_str().unwrap());
        config.add_extension("erl");

        config
            .constants
            .insert("tests".to_owned(), lit_dir.to_str().unwrap().to_string());
        config
            .constants
            .insert("lumen".to_owned(), lumen_exe.to_str().unwrap().to_string());
    })
    .expect("lit tests failed, see output for details")
}
