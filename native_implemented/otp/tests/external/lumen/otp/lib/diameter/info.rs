//! https://github.com/lumen/otp/tree/lumen/lib/diameter/src/info

use super::*;

test_compiles_lumen_otp!(diameter_dbg);
test_compiles_lumen_otp!(diameter_info);

fn relative_directory_path() -> PathBuf {
    super::relative_directory_path().join("info")
}
