//! https://github.com/lumen/otp/tree/lumen/lib/eldap/src

use std::process::Command;
use std::time::Duration;

use super::*;

test_compiles_lumen_otp!(eldap);

fn includes() -> Vec<&'static str> {
    let mut includes = super::includes();
    includes.extend(vec!["lib/eldap/ebin", "lib/eldap/include"]);

    includes
}

fn relative_directory_path() -> PathBuf {
    super::relative_directory_path().join("eldap/src")
}

fn setup() {
    let working_directory = lumen_otp_directory().join("lib/eldap/src");

    let mut command = Command::new("make");
    command
        .current_dir(&working_directory)
        .arg("../ebin/ELDAPv3.hrl");

    if let Err((command, output)) = test::timeout(
        "make ../ebin/ELDAPv3.hrl",
        working_directory.clone(),
        command,
        Duration::from_secs(10),
    ) {
        test::command_failed(
            "make ../ebin/ELDAPv3.hrl",
            working_directory,
            command,
            output,
        )
    }
}
