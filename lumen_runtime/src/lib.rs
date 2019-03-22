// For allocating multiple contiguous terms, like for Tuples.
#![feature(allocator_api)]
#![feature(duration_as_u128)]
#![feature(exact_size_is_empty)]
// For `lumen_runtime::reference::count
#![feature(integer_atomics)]
// For `lumen_runtime::binary::heap::<Iter as Iterator>.size_hint`
#![feature(ptr_offset_from)]
// for conversion of `usize` to `lumen_runtime::term::Tag`
#![feature(try_from)]
// For allocation multiple contiguous terms in `Term::alloc_count`.
#![feature(try_reserve)]
// for `lumen_runtime::term::Term`
#![feature(untagged_unions)]

#[macro_use]
extern crate cfg_if;

#[macro_use]
mod process;
#[macro_use]
mod exception;
#[macro_use]
mod support;

mod atom;
mod binary;
mod config;
mod environment;
mod float;
mod integer;
mod list;
mod logging;
mod map;
mod otp;
mod reference;
mod system;
mod term;
mod time;
mod tuple;

use self::config::Config;
use self::logging::Logger;
use self::system::break_handler;

use bus::Bus;
use log::Level;

cfg_if! {
  if #[cfg(target_arch = "wasm32")] {
    use wasm_bindgen::prelude::*;

    const NAME: &'static str = env!("CARGO_PKG_NAME");
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    #[wasm_bindgen(start)]
    pub fn start() {
      main(NAME, VERSION, std::env::args().collect());
    }
  } else {
    #[no_mangle]
    pub extern "C" fn start(name: *const libc::c_char, version: *const libc::c_char) -> i32 {
       let name = c_str_to_str!(name);
       let version = c_str_to_str!(version);
       main(name, version, std::env::args().collect());
       0
    }
  }
}

/// The main entry point for the runtime, it is invoked by the platform-specific shims found above
pub fn main(name: &str, version: &str, argv: Vec<String>) {
    // Load configuration
    let _config = Config::from_argv(name.to_string(), version.to_string(), argv)
        .expect("Could not load config!");

    // This bus is used to receive signals across threads in the system
    let mut bus: Bus<break_handler::Signal> = Bus::new(1);
    // Each thread needs a reader
    let mut rx1 = bus.add_rx();
    // Initialize the break handler with the bus, which will broadcast on it
    break_handler::init(bus);

    // Start logger
    Logger::init(Level::Info).expect("Unexpected failure initializing logger");

    // TEMP: Blocking loop which waits for user input
    loop {
        match rx1.recv() {
            Ok(_) => {
                break;
            }
            Err(e) => {
                println!("{}", e);
                break;
            }
        }
    }
}
