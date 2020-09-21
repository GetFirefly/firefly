//! https://github.com/lumen/otp/tree/lumen/erts

use super::*;

#[path = "erts/preloaded.rs"]
mod preloaded;

fn relative_directory_path() -> PathBuf {
    "erts".into()
}
