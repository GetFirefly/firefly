//! https://github.com/lumen/otp/tree/lumen/lib/crypto/src

use super::*;

test_compiles_lumen_otp!(crypto);
test_compiles_lumen_otp!(crypto_ec_curves);

fn relative_directory_path() -> PathBuf {
    super::relative_directory_path().join("crypto/src")
}
