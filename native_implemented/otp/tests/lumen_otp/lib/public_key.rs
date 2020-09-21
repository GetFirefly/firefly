//! https://github.com/lumen/otp/tree/lumen/lib/public_key/src

use std::process::Command;
use std::time::Duration;

use super::*;

test_compiles_lumen_otp!(pubkey_cert);
test_compiles_lumen_otp!(pubkey_cert_records);
test_compiles_lumen_otp!(pubkey_crl);
test_compiles_lumen_otp!(pubkey_pbe);
test_compiles_lumen_otp!(pubkey_pem);
test_compiles_lumen_otp!(pubkey_ssh);
test_compiles_lumen_otp!(public_key);

fn includes() -> Vec<&'static str> {
    let mut includes = super::includes();
    includes.extend(vec!["lib/public_key/include", "lib/public_key/src"]);

    includes
}

fn relative_directory_path() -> PathBuf {
    super::relative_directory_path().join("public_key/src")
}

pub fn setup() {
    let working_directory = lumen_otp_directory().join("lib/public_key/asn1");

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
