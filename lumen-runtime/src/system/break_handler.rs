use std::sync::mpsc;
use std::thread;

use bus::Bus;

#[derive(Clone)]
pub enum Signal {
    TERM,
    QUIT,
    HUP,
    ABRT,
    ALRM,
    USR1,
    USR2,
    CHLD,
    STOP,
}

// Signal handling doesn't apply to WebAssembly
#[cfg(target_arch = "wasm32")]
pub fn init(mut bus: Bus<Signal>) -> Result<bool, std::io::Error> {
    Ok(false)
}

// But should everywhere else
#[cfg(not(target_arch = "wasm32"))]
pub fn init(mut bus: Bus<Signal>) -> Result<bool, std::io::Error> {
    use signal_hook::*;

    // Fan-in for dispatcher
    let (tx, rx) = mpsc::sync_channel(10);
    // Dispatcher thread which relays signals to receivers
    thread::spawn(move || {
        for signal in rx.iter() {
            bus.broadcast(signal);
        }
    });
    // Signal mapping from library to internal enum
    let signals = vec![
        (SIGTERM, Signal::TERM),
        (SIGQUIT, Signal::QUIT),
        (SIGHUP, Signal::HUP),
        (SIGABRT, Signal::ABRT),
        (SIGALRM, Signal::ALRM),
        (SIGUSR1, Signal::USR1),
        (SIGUSR2, Signal::USR2),
        (SIGCHLD, Signal::CHLD),
        (SIGSTOP, Signal::STOP),
    ];
    // For each pair, register a handler which sends the signal to the dispatcher
    for (hook, signal) in signals {
        let stx = tx.clone();
        unsafe {
            signal_hook::register(hook, move || {
                stx.send(signal.clone()).unwrap_or_default();
            })?;
        }
    }
    Ok(true)
}
