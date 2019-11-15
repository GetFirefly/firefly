#![deny(warnings)]
#![allow(stable_features)]
// `rand` has link errors
#![allow(intra_doc_link_resolution_failure)]
// For allocating multiple contiguous terms, like for Tuples.
#![feature(allocator_api)]
#![feature(bind_by_move_pattern_guards)]
#![feature(exact_size_is_empty)]
// For `lumen_runtime::otp::erlang::term_to_binary`
#![feature(float_to_from_bytes)]
#![feature(fn_traits)]
// For `lumen_runtime::reference::count
#![feature(integer_atomics)]
// For `lumen_runtime::binary::heap::<Iter as Iterator>.size_hint`
#![feature(ptr_offset_from)]
// For allocation multiple contiguous terms in `Term::alloc_count`.
#![feature(try_reserve)]
#![feature(type_ascription)]
// for `lumen_runtime::term::Term`
#![feature(untagged_unions)]
// for `lumen_runtime::distribution::nodes::insert`
#![feature(option_unwrap_none)]
// for `lumen_runtime::list::Cons::subtract`.
#![feature(vec_remove_item)]
// `crate::registry::<Registered as PartialEq>::eq`
#![feature(weak_ptr_eq)]
// Layout helpers
#![feature(alloc_layout_extra)]

extern crate alloc;
#[macro_use]
extern crate cfg_if;
#[macro_use]
extern crate lazy_static;

extern crate chrono;

#[macro_use]
mod macros;

mod binary;
pub mod binary_to_string;
// `pub` or `examples/spawn-chain`
pub mod code;
mod config;
mod distribution;
pub mod future;
mod logging;
mod number;
pub mod otp;
pub mod process;
// `pub` or `examples/spawn-chain`
pub mod registry;
mod run;
// `pub` for `examples/spawn-chain`
pub mod scheduler;
mod send;
mod stacktrace;
// `pub` for `examples/spawn-chain`
pub mod system;
// `pub` for `examples/spawn-chain`
mod term;
// `pub` to allow `time::monotonic::set_source(callback)`
#[cfg(test)]
mod test;
pub mod time;
// Public so that external code can all `timer::expire` to expire timers
mod timer;

use self::config::Config;
use self::logging::Logger;
use self::system::break_handler;

use bus::Bus;
use log::Level;

cfg_if! {
  if #[cfg(target_arch = "wasm32")] {
//    use wasm_bindgen::prelude::*;
//
//    const NAME: &'static str = env!("CARGO_PKG_NAME");
//    const VERSION: &'static str = env!("CARGO_PKG_VERSION");
//
//    #[wasm_bindgen(start)]
//    pub fn start() {
//      main(NAME, VERSION, std::env::args().collect());
//    }
  } else {
    #[no_mangle]
    pub extern "C" fn start(name: *const libc::c_char, version: *const libc::c_char) -> i32 {
       let name = c_str_to_str!(name);
       let version = c_str_to_str!(version);
       match main(name, version, std::env::args().collect()) {
           Ok(_) => 0,
           Err(err) => {
               println!("{:?}", err);
               1
           }
       }
    }
  }
}

/// The main entry point for the runtime, it is invoked by the platform-specific shims found above
pub fn main(name: &str, version: &str, argv: Vec<String>) -> anyhow::Result<()> {
    // Load configuration
    let _config = Config::from_argv(name.to_string(), version.to_string(), argv)?;

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
        let _ = rx1.recv()?;
    }
}
