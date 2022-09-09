extern crate log;

use std::env;
use std::process;
use std::time::Instant;

use anyhow::{anyhow, bail};

use firefly_compiler as driver;
use firefly_util::time;

pub fn main() -> anyhow::Result<()> {
    // Handle unexpected panics by presenting a user-friendly bug report prompt;
    // except when we're requesting debug info from the compiler explicitly, in
    // which case we don't want to hide the panic
    if env::var_os("FIREFLY_LOG").is_none() {
        human_panic::setup_panic!();
    }

    // Initialize logger
    let mut builder = env_logger::Builder::from_env("FIREFLY_LOG");
    builder.format_indent(Some(2));
    if let Ok(precision) = env::var("FIREFLY_LOG_WITH_TIME") {
        match precision.as_str() {
            "s" => builder.format_timestamp_secs(),
            "ms" => builder.format_timestamp_millis(),
            "us" => builder.format_timestamp_micros(),
            "ns" => builder.format_timestamp_nanos(),
            other => bail!(
                "invalid FIREFLY_LOG_WITH_TIME precision, expected one of [s, ms, us, ns], got '{}'",
                other
            ),
        };
    } else {
        builder.format_timestamp(None);
    }
    builder.init();

    // Get current working directory
    let cwd = env::current_dir().map_err(|e| anyhow!("Current directory is invalid: {}", e))?;

    // Get the current instant, in case needed for timing later
    let print_timings = env::var("FIREFLY_TIMING").is_ok();
    let start = Instant::now();

    // Run compiler
    match driver::run_compiler(cwd, env::args_os()) {
        Ok(status_code) => {
            time::print_time_passes_entry(print_timings, "\ttotal", start.elapsed());
            process::exit(status_code);
        }
        Err(err) => {
            time::print_time_passes_entry(print_timings, "\ttotal", start.elapsed());
            if let Some(err) = err.downcast_ref::<clap::Error>() {
                err.exit()
            } else {
                eprintln!("{}", err);
                process::exit(1);
            }
        }
    }
}
