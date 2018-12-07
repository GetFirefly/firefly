mod config;
mod logging;
mod system;

use std::ffi::CStr;

use self::config::Config;
use self::logging::Logger;
use self::system::break_handler;

use bus::Bus;
use internment::Intern;
use log::Level;

extern "C" {
    static APP_NAME: *const libc::c_char;
    static APP_VERSION: *const libc::c_char;
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn start() {
    main(Vec::new());
}

#[cfg(not(target_arch = "wasm32"))]
#[no_mangle]
pub extern "C" fn start() {
    main(Vec::new());
}

pub fn main(argv: Vec<String>) {
    // Load configuration
    let app_name = unsafe { CStr::from_ptr(APP_NAME).to_string_lossy().into_owned() };
    let app_version = unsafe { CStr::from_ptr(APP_VERSION).to_string_lossy().into_owned() };
    let _config = Config::from_argv(app_name, app_version, argv).expect("Could not load config!");

    // Initialize break handler
    let bus: Bus<break_handler::Signal> = Bus::new(1);
    //Get reader
    //let rx1 = bus.add_rx();
    //Read event: rx1.try_recv() = Result<Signal, mpsc::TryRecvError>
    break_handler::init(bus).expect("Unexpected failure initializing signal handling");

    // Start logger
    Logger::init(Level::Info).expect("Unexpected failure initializing logger");

    //TODO: initialize atom table
    //To create/lookup: let sym = Intern::new(a)
    for a in vec!["atom"] {
        Intern::new(a);
    }

    //TODO: initiaiize ETS
    //TODO: initialize scheduler
    //TODO: start other scheduler threads (if needed)
    //TODO: enter scheduler loop, start init process, enter via otp_ring0
    //TODO: we're now shutting down, clean up, exit
}
