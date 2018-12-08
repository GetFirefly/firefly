mod config;
mod logging;
mod system;
#[macro_use] mod support;

use self::config::Config;
use self::logging::Logger;
use self::system::break_handler;

use bus::Bus;
use log::Level;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn start(name: &str, version: &str) {
    main(name, version, std::env::args().collect());
}

#[cfg(not(target_arch = "wasm32"))]
#[no_mangle]
pub extern "C" fn start(name: *const libc::c_char, version: *const libc::c_char) -> i32 {
    let name = c_str_to_str!(name);
    let version = c_str_to_str!(version);
    main(name, version, std::env::args().collect());
    0
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
            },
            Err(e) => {
                println!("{}", e);
                break;
            }
        }
    }
}
