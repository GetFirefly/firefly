//! https://github.com/lumen/otp/tree/lumen/lib/snmp/src

#[path = "snmp/agent.rs"]
mod agent;
#[path = "snmp/app.rs"]
mod app;
#[path = "snmp/compile.rs"]
mod compile;
#[path = "snmp/manager.rs"]
mod manager;
#[path = "snmp/mibs.rs"]
mod mibs;
#[path = "snmp/misc.rs"]
mod misc;

use super::*;

fn relative_directory_path() -> PathBuf {
    super::relative_directory_path().join("snmp/src")
}
