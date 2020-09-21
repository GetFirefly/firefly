//! https://github.com/lumen/otp/tree/lumen/lib/ftp/src

use super::*;

test_compiles_lumen_otp!(ftp);
test_compiles_lumen_otp!(ftp_app imports "lib/ftp/src/ftp_sup");
test_compiles_lumen_otp!(ftp_progress);
test_compiles_lumen_otp!(ftp_response);
test_compiles_lumen_otp!(ftp_sup);

fn includes() -> Vec<&'static str> {
    let mut includes = super::includes();
    includes.push("lib/ftp/src");

    includes
}

fn relative_directory_path() -> PathBuf {
    super::relative_directory_path().join("ftp/src")
}
