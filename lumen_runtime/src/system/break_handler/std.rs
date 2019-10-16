use std::thread;

use bus::Bus;

use super::Signal;

impl From<usize> for Signal {
    fn from(sig: usize) -> Signal {
        match sig as libc::c_int {
            signal_hook::SIGINT => Signal::INT,
            signal_hook::SIGTERM => Signal::TERM,
            signal_hook::SIGQUIT => Signal::QUIT,
            signal_hook::SIGHUP => Signal::HUP,
            signal_hook::SIGABRT => Signal::ABRT,
            signal_hook::SIGALRM => Signal::ALRM,
            signal_hook::SIGUSR1 => Signal::USR1,
            signal_hook::SIGUSR2 => Signal::USR2,
            signal_hook::SIGCHLD => Signal::CHLD,
            _ => Signal::Unknown,
        }
    }
}

pub fn init(mut bus: Bus<Signal>) {
    thread::spawn(move || {
        use signal_hook::iterator::Signals;

        let signals = Signals::new(&[
            signal_hook::SIGINT,
            signal_hook::SIGTERM,
            signal_hook::SIGQUIT,
            signal_hook::SIGHUP,
            signal_hook::SIGABRT,
            signal_hook::SIGALRM,
            signal_hook::SIGUSR1,
            signal_hook::SIGUSR2,
            signal_hook::SIGCHLD,
        ])
        .expect("could not bind signal handlers");

        for signal in signals.forever() {
            match Signal::from(signal as usize) {
                Signal::Unknown => (),
                sig => bus.broadcast(sig),
            }
        }
    });
}
