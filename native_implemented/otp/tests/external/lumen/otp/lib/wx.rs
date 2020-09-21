//! https://github.com/lumen/otp/tree/lumen/lib/wx/src

use super::*;

#[path = "wx/gen.rs"]
mod gen;

test_compiles_lumen_otp!(wx);
test_compiles_lumen_otp!(wx_object);
test_compiles_lumen_otp!(wxe_master);
test_compiles_lumen_otp!(wxe_server);
test_compiles_lumen_otp!(wxe_util);

fn includes() -> Vec<&'static str> {
    let mut includes = super::includes();
    includes.extend(vec!["lib/wx/src/"]);

    includes
}

fn relative_directory_path() -> PathBuf {
    super::relative_directory_path().join("wx/src")
}
