//! https://github.com/lumen/otp/tree/lumen/lib/inets/src

#[path = "inets/http_client.rs"]
mod http_client;
#[path = "inets/http_lib.rs"]
mod http_lib;
#[path = "inets/http_server.rs"]
mod http_server;
#[path = "inets/inets_app.rs"]
mod inets_app;

use super::*;

fn relative_directory_path() -> PathBuf {
    super::relative_directory_path().join("inets/src")
}
