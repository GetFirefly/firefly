use std::thread;

use bus::Bus;

use super::Signal;

impl From<usize> for Signal {
    fn from(sig: usize) -> Signal {
        match sig as libc::c_int {
            signal_hook::consts::SIGINT => Signal::INT,
            signal_hook::consts::SIGTERM => Signal::TERM,
            signal_hook::consts::SIGQUIT => Signal::QUIT,
            signal_hook::consts::SIGHUP => Signal::HUP,
            signal_hook::consts::SIGABRT => Signal::ABRT,
            signal_hook::consts::SIGALRM => Signal::ALRM,
            signal_hook::consts::SIGUSR1 => Signal::USR1,
            signal_hook::consts::SIGUSR2 => Signal::USR2,
            signal_hook::consts::SIGCHLD => Signal::CHLD,
            _ => Signal::Unknown,
        }
    }
}

pub fn init(mut bus: Bus<Signal>) {
    thread::spawn(move || {
        use signal_hook::iterator::Signals;

        let mut signals = Signals::new(&[
            signal_hook::consts::SIGINT,
            signal_hook::consts::SIGTERM,
            signal_hook::consts::SIGQUIT,
            signal_hook::consts::SIGHUP,
            signal_hook::consts::SIGABRT,
            signal_hook::consts::SIGALRM,
            signal_hook::consts::SIGUSR1,
            signal_hook::consts::SIGUSR2,
            signal_hook::consts::SIGCHLD,
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
