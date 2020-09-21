//! https://github.com/lumen/otp/tree/lumen/lib/tftp/src

use super::*;

test_compiles_lumen_otp!(tftp imports "lib/kernel/src/application", "lib/tftp/src/tftp_engine", "lib/tftp/src/tftp_sup");
test_compiles_lumen_otp!(tftp_app imports "lib/tftp/src/tftp_sup");
test_compiles_lumen_otp!(tftp_binary);
test_compiles_lumen_otp!(tftp_engine);
test_compiles_lumen_otp!(tftp_file);
test_compiles_lumen_otp!(tftp_lib);
test_compiles_lumen_otp!(tftp_logger imports "lib/kernel/src/error_logger");
test_compiles_lumen_otp!(tftp_sup);

fn includes() -> Vec<&'static str> {
    let mut includes = super::includes();
    includes.extend(vec!["lib/kernel/src", "lib/tftp/src"]);

    includes
}

fn relative_directory_path() -> PathBuf {
    super::relative_directory_path().join("tftp/src")
}
