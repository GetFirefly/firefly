//! https://github.com/lumen/otp/tree/lumen/lib/snmp/src/mibs

use std::process::Command;
use std::time::Duration;

use super::*;

pub fn setup() {
    let working_directory = lumen_otp_directory().join("lib/snmp/mibs");

    let mut command = Command::new("make");
    command.current_dir(&working_directory);

    if let Err((command, output)) = test::timeout(
        "make",
        working_directory.clone(),
        command,
        Duration::from_secs(10),
    ) {
        test::command_failed("make", working_directory, command, output)
    }
}
