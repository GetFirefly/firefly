#![feature(asm)]
#![feature(naked_functions)]
#![feature(termination_trait_lib)]
#![feature(thread_local)]
#![feature(alloc_layout_extra)]

#[cfg(not(unix))]
compile_error!("lumen_rt_minimal is only supported on unix targets!");

#[macro_use]
mod macros;
mod config;
mod logging;
mod scheduler;
mod sys;
mod distribution;
mod process;

use bus::Bus;
use log::Level;
use std::thread;

use lumen_rt_core as rt_core;

use self::config::Config;
use self::scheduler::Scheduler;
use self::sys::break_handler::{self, Signal};

#[liblumen_core::entry]
fn main() -> impl ::std::process::Termination + 'static {
    let name = env!("CARGO_PKG_NAME");
    let version = env!("CARGO_PKG_VERSION");
    main_internal(name, version, Vec::new())
}

fn main_internal(name: &str, version: &str, argv: Vec<String>) -> Result<(), ()> {
    // Load system configuration
    let _config = match Config::from_argv(name.to_string(), version.to_string(), argv) {
        Ok(config) => config,
        Err(err) => {
            panic!("Config error: {}", err);
            return Err(());
        }
    };

    // This bus is used to receive signals across threads in the system
    let mut bus: Bus<break_handler::Signal> = Bus::new(1);
    // Each thread needs a reader
    let mut rx1 = bus.add_rx();
    // Initialize the break handler with the bus, which will broadcast on it
    break_handler::init(bus);

    // Start logger
    let level_filter = Level::Info.to_level_filter();
    logging::init(level_filter).expect("Unexpected failure initializing logger");

    liblumen_alloc::erts::apply::dump_symbols();
    liblumen_alloc::erts::term::atom::dump_atoms();
    let scheduler = <Scheduler as rt_core::Scheduler>::current();
    scheduler.init().unwrap();
    loop {
        // Run the scheduler for a cycle
        let scheduled = scheduler.run_once();
        // Check for system signals, and terminate if needed
        if let Ok(sig) = rx1.try_recv() {
            match sig {
                // For now, SIGINT initiates a controlled shutdown
                Signal::INT => {
                    // If an error occurs, report it before shutdown
                    if let Err(err) = scheduler.shutdown() {
                        eprintln!("System error: {}", err);
                        return Err(());
                    } else {
                        break;
                    }
                }
                // Technically, we may never see these signals directly,
                // we may just be terminated out of hand; but just in case,
                // we handle them explicitly by immediately terminating, so
                // that we are good citizens of the operating system
                sig if sig.should_terminate() => {
                    return Err(());
                }
                // All other signals can be surfaced to other parts of the
                // system for custom use, e.g. SIGCHLD, SIGALRM, SIGUSR1/2
                _ => (),
            }
        }
        // If the scheduler scheduled a process this cycle, then we're busy
        // and should keep working until we have an idle period
        if scheduled {
            continue;
        }
        // Otherwise,
        // In some configurations, it makes more sense for us to spin and use
        // spin_loop_hint here instead; namely when we're supposed to be the primary
        // software on a system, and threads are pinned to cores, it makes no sense
        // to yield to the system scheduler. However on a system with contention for
        // system resources, or where threads aren't pinned to cores, we're better off
        // explicitly yielding to the scheduler, rather than waiting to be preempted at
        // a potentially inopportune time.
        //
        // In any case, for now, we always explicitly yield until we've got proper support
        // for configuring the system
        thread::yield_now()
    }

    Ok(())
}
