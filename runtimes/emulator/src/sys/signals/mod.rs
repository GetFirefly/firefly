mod break_handler;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;

use signal_hook::consts::signal::*;
use signal_hook::consts::TERM_SIGNALS;
use signal_hook::flag;
use signal_hook::iterator::Signals;

use firefly_alloc::fragment::HeapFragment;
use firefly_alloc::heap::Heap;
use firefly_rt::services::registry::{self, Registrant, WeakAddress};
use firefly_rt::term::{atoms, Atom, LayoutBuilder, Term, TermFragment, Tuple};

use smallvec::SmallVec;

#[cfg(not(windows))]
const ALLOWED_SIGNALS: &'static [libc::c_int] =
    &[SIGHUP, SIGUSR1, SIGUSR2, SIGCHLD, SIGTSTP, SIGABRT, SIGALRM];

#[cfg(windows)]
const ALLOWED_SIGNALS: &'static [libc::c_int] = &[SIGBREAK];

/// This starts the signal dispatcher loop.
///
/// The signal dispatcher is responsible for responding quickly
/// and efficiently to incoming signals sent to the application.
///
/// It must never block, i.e. it should dispatch work to other threads
/// rather than try to handle it in this core loop. This ensures that
/// we can always break twice to interrupt the system regardless of what
/// else is going on.
#[cfg(not(windows))]
pub fn start_handler() {
    let shutdown = Arc::new(AtomicBool::new(false));
    let break_requested = Arc::new(AtomicBool::new(false));

    for sig in TERM_SIGNALS {
        // When terminated by a second signal, exit with code 1.
        //
        // This only takes effect on the second signal
        flag::register_conditional_shutdown(*sig, 1, Arc::clone(&shutdown)).unwrap();
        // Prepare the conditional shutdown above by setting the flag above
        // to true when first receiving a signal
        flag::register(*sig, Arc::clone(&shutdown)).unwrap();
    }

    let break_handler_shutdown = Arc::clone(&shutdown);
    let break_handler_requested = Arc::clone(&break_requested);
    let break_handler = thread::Builder::new()
        .name("break_handler".into())
        .spawn(move || break_handler::run(break_handler_shutdown, break_handler_requested))
        .unwrap();

    // Subscribe to the following signals, including information about their origin
    let mut sigs = SmallVec::<[libc::c_int; 8]>::new();
    // Generic signal handling
    sigs.extend_from_slice(ALLOWED_SIGNALS);
    sigs.extend_from_slice(TERM_SIGNALS);
    let mut signals = Signals::new(&sigs).unwrap();

    for signal in signals.forever() {
        match signal {
            SIGINT => {
                break_requested.store(true, Ordering::Release);
                break_handler.thread().unpark();
            }
            SIGUSR1 => signal_notify_requested(atoms::Sigusr1),
            SIGUSR2 => signal_notify_requested(atoms::Sigusr2),
            SIGCHLD => signal_notify_requested(atoms::Sigchld),
            SIGSTOP => signal_notify_requested(atoms::Sigstop),
            SIGTSTP => signal_notify_requested(atoms::Sigtstp),
            SIGQUIT => signal_notify_requested(atoms::Sigquit),
            SIGTERM => signal_notify_requested(atoms::Sigterm),
            SIGHUP => signal_notify_requested(atoms::Sighup),
            SIGABRT => signal_notify_requested(atoms::Sigabrt),
            SIGALRM => signal_notify_requested(atoms::Sigalrm),
            _ => (), // ignore
        }
    }

    break_handler.thread().unpark();
    break_handler.join().unwrap();
}

#[cfg(windows)]
pub fn start_handler() {
    todo!()
}

/// Send `{notify, Signal}` to `erl_signal_server` process
#[inline(never)]
fn signal_notify_requested(signal: Atom) {
    if let Some(Registrant::Process(proc)) = registry::get_by_name(atoms::ErlSignalServer) {
        let mut locked = proc.lock();
        let message = {
            let mut builder = LayoutBuilder::new();
            builder.build_tuple(2);
            let layout = builder.finish();
            if locked.heap_available() >= layout.size() {
                let term = Term::Tuple(
                    Tuple::from_slice(&[atoms::Notify.into(), signal.into()], &mut locked).unwrap(),
                );
                TermFragment {
                    term: term.into(),
                    fragment: None,
                }
            } else {
                let fragment_ptr = HeapFragment::new(layout, None).unwrap();
                let fragment = unsafe { fragment_ptr.as_ref() };
                let term = Term::Tuple(
                    Tuple::from_slice(&[atoms::Notify.into(), signal.into()], fragment).unwrap(),
                );
                TermFragment {
                    term: term.into(),
                    fragment: Some(fragment_ptr),
                }
            }
        };

        locked.send_fragment(WeakAddress::System, message).ok();
    }
}
