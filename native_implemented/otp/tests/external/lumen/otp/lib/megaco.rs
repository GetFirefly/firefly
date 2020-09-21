//! https://github.com/lumen/otp/tree/lumen/lib/megaco/src

#[path = "megaco/app.rs"]
mod app;
#[path = "megaco/binary.rs"]
mod binary;
#[path = "megaco/engine.rs"]
mod engine;
#[path = "megaco/flex.rs"]
mod flex;
#[path = "megaco/tcp.rs"]
mod tcp;
#[path = "megaco/text.rs"]
mod text;
#[path = "megaco/udp.rs"]
mod udp;

use super::*;

fn relative_directory_path() -> PathBuf {
    super::relative_directory_path().join("megaco/src")
}
