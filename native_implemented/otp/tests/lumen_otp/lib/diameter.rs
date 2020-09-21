//! https://github.com/lumen/otp/tree/lumen/lib/diameter/src

#[path = "diameter/base.rs"]
mod base;
#[path = "diameter/compiler.rs"]
mod compiler;
#[path = "diameter/info.rs"]
mod info;
#[path = "diameter/transport.rs"]
mod transport;

use super::*;

fn relative_directory_path() -> PathBuf {
    super::relative_directory_path().join("diameter/src")
}
