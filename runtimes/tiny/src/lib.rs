#![feature(c_unwind)]
#![feature(once_cell)]
#![feature(ptr_metadata)]
#![feature(process_exitcode_internals)]
#![feature(thread_local)]
#![feature(let_else)]

extern crate liblumen_crt;

mod env;
mod erlang;
mod init;
mod intrinsic;
mod scheduler;
mod sys;

use anyhow::anyhow;
use bus::Bus;

use self::sys::break_handler::{self, Signal};

#[export_name = "lumen_entry"]
pub unsafe extern "C" fn main() -> i32 {
    use std::process::Termination;

    let name = env!("CARGO_PKG_NAME");
    let version = env!("CARGO_PKG_VERSION");
    main_internal(name, version, Vec::new()).report().to_i32()
}

fn main_internal(_name: &str, _version: &str, _argv: Vec<String>) -> anyhow::Result<()> {
    self::env::init(std::env::args_os()).unwrap();

    // This bus is used to receive signals across threads in the system
    let mut bus: Bus<Signal> = Bus::new(1);
    // Each thread needs a reader
    let mut rx1 = bus.add_rx();
    // Initialize the break handler with the bus, which will broadcast on it
    break_handler::init(bus);

    scheduler::init();
    scheduler::with_current(|scheduler| scheduler.spawn_init()).unwrap();
    loop {
        // Run the scheduler for a cycle
        let scheduled = scheduler::with_current(|scheduler| scheduler.run_once());
        // Check for system signals, and terminate if needed
        if let Ok(sig) = rx1.try_recv() {
            match sig {
                // For now, SIGINT initiates a controlled shutdown
                Signal::INT => {
                    // If an error occurs, report it before shutdown
                    if let Err(err) = scheduler::with_current(|s| s.shutdown()) {
                        return Err(anyhow!(err));
                    } else {
                        break;
                    }
                }
                // Technically, we may never see these signals directly,
                // we may just be terminated out of hand; but just in case,
                // we handle them explicitly by immediately terminating, so
                // that we are good citizens of the operating system
                sig if sig.should_terminate() => {
                    return Ok(());
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

        break;
    }

    match scheduler::with_current(|s| s.shutdown()) {
        Ok(_) => Ok(()),
        Err(err) => Err(anyhow!(err)),
    }
}
