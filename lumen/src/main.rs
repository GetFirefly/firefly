extern crate log;

use std::env;
use std::process;
use std::time::Instant;

use anyhow::anyhow;

use liblumen_compiler::{self as driver, argparser};
use liblumen_session::ShowOptionGroupHelp;
use liblumen_util::error::HelpRequested;
use liblumen_util::time;

pub fn main() -> anyhow::Result<()> {
    // Handle unexpected panics by presenting a user-friendly bug report prompt
    //human_panic::setup_panic!();

    // Initialize logger
    env_logger::init_from_env("LUMEN_LOG");

    // Get the current instant, in case needed for timing later
    let start = Instant::now();

    // Get current working directory
    let cwd = env::current_dir().map_err(|e| anyhow!("Current directory is invalid: {}", e))?;

    // Run compiler
    if let Err(err) = driver::run_compiler(cwd, env::args_os()) {
        if let Some(err) = err.downcast_ref::<HelpRequested>() {
            handle_help(err);
        }
        if let Some(err) = err.downcast_ref::<ShowOptionGroupHelp>() {
            handle_option_group_help(err);
        }
        if let Some(err) = err.downcast_ref::<clap::Error>() {
            handle_clap_err(err);
        }
        eprintln!("{}", err);
        process::exit(1);
    }

    let print_timings = env::var("LUMEN_TIMING").is_ok();
    time::print_time_passes_entry(print_timings, "\ttotal", start.elapsed());

    Ok(())
}

fn handle_help(err: &HelpRequested) -> ! {
    match err.primary() {
        "compile" => argparser::print_compile_help(),
        "print" => argparser::print_print_help(),
        _ => unimplemented!(),
    }
    process::exit(1);
}

fn handle_option_group_help(err: &ShowOptionGroupHelp) -> ! {
    err.print_help();
    process::exit(0);
}

fn handle_clap_err(err: &clap::Error) -> ! {
    err.exit()
}
