use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;

/// Starts the break handler loop.
///
/// The provided references to atomic flags are used to check whether the
/// break handler was requested to process a break signal - or if it is
/// being asked to terminate/shut down.
pub fn run(shutdown: Arc<AtomicBool>, break_requested: Arc<AtomicBool>) {
    loop {
        thread::park();

        // Check if we were woken up to shut down
        if !break_requested.swap(false, Ordering::Acquire) {
            break;
        }

        // We were woken up to execute the break handler
        handle_break();

        // Reset the shutdown flag since we handled this break
        shutdown.store(false, Ordering::SeqCst);
    }
}

const HELP: &'static str = r#"
BREAK: (a)bort (A)bort with dump (c)ontinue (p)roc info (i)info
       (l)oaded (v)ersion (k)ill (D)b-tables (d)istribution
"#;

#[cfg(not(windows))]
fn handle_break() {
    use std::io::{IsTerminal, Write};

    const CLEARSCREEN: &'static [u8] = b"\x1b[J";

    {
        let mut stdout = io::stdout().lock();

        if !io::stdin().is_terminal() || !stdout.is_terminal() {
            stdout.write_all(b"\n").unwrap();
            stdout.write_all(HELP.as_bytes()).unwrap();
        } else {
            stdout.write_all(b"\n").unwrap();
            stdout.write_all(CLEARSCREEN).unwrap();
            stdout.write_all(HELP.as_bytes()).unwrap();
        }
        stdout.flush().unwrap();
    }

    loop {
        if let Ok(key) = get_key() {
            match key {
                'q' | 'a' | '*' => {
                    // Abort
                    // The asterisk is a read error on windows, treat it as 'a' in this case
                    std::process::exit(0);
                }
                'A' => {
                    // Abort with crash dump
                    eprintln!("Crash dump requested by user");
                    // TODO: generate crash dump
                    std::process::exit(-4);
                }
                'c' => {
                    // Continue
                    return;
                }
                'p' => {
                    // Print process info
                    println!("Information about the active processes is not implemented yet");
                    return;
                }
                'o' => {
                    // Print port info
                    println!("Information about the active ports is not implemented yet");
                    return;
                }
                'i' => {
                    // Print system info
                    println!("Information about the current runtime system is not implemented yet");
                    return;
                }
                'l' => {
                    // Print loaded info
                    println!(
                        "Information about loaded applications/modules is not implemented yet"
                    );
                    return;
                }
                'v' => {
                    // Print version info
                    println!("Erlang (Firefly) emulator version 1.0");
                    return;
                }
                'd' => {
                    // Print distribution info
                    println!("=node:'nonode@nohost'\n=no_distribution");
                    return;
                }
                'D' => {
                    // Print ETS database info
                    println!("ETS is not available in this system");
                    return;
                }
                'k' => {
                    // Process killer
                    println!("The process killer is not implemented yet");
                    return;
                }
                '\n' => continue,
                c => {
                    print!(
                        "Invalid option '{}'. Please enter one of the following:\n{}",
                        c, HELP
                    );
                }
            }
        } else {
            std::process::exit(0);
        }
    }
}

#[cfg(windows)]
fn handle_break() {
    todo!()
}

#[cfg(not(windows))]
fn get_key() -> Result<char, ()> {
    use std::io::{ErrorKind, Read};
    use std::slice;

    let mut stdin = io::stdin().lock();

    let mut byte = 0;
    loop {
        match stdin.read(slice::from_mut(&mut byte)) {
            Ok(0) => continue,
            Ok(_) => return char::from_u32(byte as u32).ok_or(()),
            Err(ref e) if e.kind() == ErrorKind::Interrupted => continue,
            Err(_) => break,
        }
    }

    Err(())
}
