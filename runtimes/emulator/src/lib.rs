#![feature(c_unwind)]
#![feature(core_intrinsics)]
#![feature(allocator_api)]
#![feature(once_cell)]
#![feature(thread_local)]
#![feature(iterator_try_collect)]
#![feature(int_roundings)]
#![feature(assert_matches)]
#![feature(process_exitcode_internals)]
#![feature(ptr_metadata)]
#![feature(is_terminal)]
#![feature(option_result_contains)]
#![feature(const_trait_impl)]
#![feature(const_default_impls)]
#![feature(slice_as_chunks)]
#![feature(local_key_cell_methods)]
#![feature(box_into_inner)]

extern crate firefly_crt;

mod bifs;
mod emulator;
mod nifs;
mod queue;
mod sys;
mod unique;

use std::convert::Infallible;
use std::env;
use std::panic;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use crossbeam::deque::Injector;

use firefly_bytecode::{ByteCode, BytecodeReader, ReadError};
use firefly_rt::scheduler;
use firefly_rt::services::{self, distribution::NoDistribution};
use firefly_rt::term::{atom::GlobalAtomTable, Atom};

use self::emulator::{Emulator, EmulatorError};

const NUM_SCHEDULERS: usize = 1;

#[macro_export]
macro_rules! badarg {
    ($process:expr, $term:expr) => {
        return {
            $process.exception_info.flags = firefly_rt::error::ExceptionFlags::ERROR;
            $process.exception_info.reason = firefly_rt::term::atoms::Badarg.into();
            $process.exception_info.value = $term;
            $process.exception_info.args = Some($term);
            $process.exception_info.trace = None;
            firefly_rt::function::ErlangResult::Err
        }
    };
}

#[macro_export]
macro_rules! unwrap_or_badarg {
    ($process:expr, $term:expr, $value:expr) => {
        match $value {
            Ok(value) => value,
            Err(_) => badarg!($process, $term),
        }
    };
}

#[export_name = "firefly_entry"]
pub fn main() -> i32 {
    use std::process::Termination;

    let mut builder = env_logger::Builder::from_env("ERTS_TRACE");
    builder.format_indent(Some(2));
    if let Ok(precision) = env::var("ERTS_TRACE_WITH_TIME") {
        match precision.as_str() {
            "s" => builder.format_timestamp_secs(),
            "ms" => builder.format_timestamp_millis(),
            "us" => builder.format_timestamp_micros(),
            "ns" => builder.format_timestamp_nanos(),
            other => {
                eprintln!("Ignoring invalid ERTS_TRACE_WITH_TIME value, expected one of [s, ms, us, ns], got '{}'. Using 'ms' instead..", other);
                builder.format_timestamp_millis()
            }
        };
    } else {
        builder.format_timestamp(None);
    }
    builder.init();

    // Load bytecode first, since if it fails there is no point in going further
    let code = load_bytecode().expect("failed to load bytecode");

    // Initialize the global environment
    sys::env::init(std::env::args_os()).unwrap();

    // Initialize global uniqueness data
    self::unique::init(NUM_SCHEDULERS, 0, 0);

    // Initialize the distribution service
    services::distribution::init(NoDistribution::new());

    // Create a new multi-threaded async runtime
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("unable to start scheduler");

    // Set up the system signal handler
    if cfg!(not(target_family = "wasm")) {
        runtime.spawn_blocking(|| sys::signals::start_handler());
    }
    // Set up the system dispatcher
    runtime.spawn(sys::dispatcher::start());
    // Get a clone of the async runtime handle to give to each scheduler
    let handle = runtime.handle().clone();
    // Get the global work-stealing task queue shared by the schedulers
    let injector = Arc::new(Injector::new());
    // Spawn a task for each instance of emulator acting as a scheduler
    let mut handles = Vec::with_capacity(NUM_SCHEDULERS);
    for i in 0..NUM_SCHEDULERS {
        let emu_handle = handle.clone();
        let emu_injector = injector.clone();
        let emu_code = code.clone();
        handles.push(runtime.spawn_blocking(move || {
            let emulator = scheduler::create(move |id| {
                Ok::<_, Infallible>(Emulator::new(id, emu_code, emu_injector, emu_handle))
            })
            .unwrap();
            let spawn_init = i == 0;
            emulator.start(spawn_init)
        }));
    }

    // Wait for all of the scheduler threads to terminate
    for handle in handles.drain(..) {
        match runtime.block_on(handle) {
            Err(join_err) => {
                if let Ok(reason) = join_err.try_into_panic() {
                    panic::resume_unwind(reason);
                }
                return ExitCode::FAILURE.report().to_i32();
            }
            Ok(result) => match result {
                Ok(_) => continue,
                Err(EmulatorError::Halt(0)) => {
                    // Give some time for any outstanding background tasks to clean up
                    runtime.shutdown_timeout(Duration::from_millis(50));

                    return ExitCode::SUCCESS.report().to_i32();
                }
                Err(EmulatorError::Halt(n)) => {
                    // Give some time for any outstanding background tasks to clean up
                    runtime.shutdown_timeout(Duration::from_secs(5));

                    return n as i32;
                }
                Err(EmulatorError::InvalidInit) => {
                    eprintln!("invalid init!");
                    return ExitCode::FAILURE.report().to_i32();
                }
                Err(EmulatorError::SystemLimit) => {
                    eprintln!("exceeded system limit, see standard error for details");
                    return ExitCode::FAILURE.report().to_i32();
                }
            },
        }
    }

    ExitCode::SUCCESS.report().to_i32()
}

fn load_bytecode() -> Result<Arc<ByteCode<Atom, GlobalAtomTable>>, ReadError<GlobalAtomTable>> {
    use core::slice;

    extern "C" {
        #[link_name = "__FIREFLY_BC_LEN"]
        static BYTECODE_LEN: usize;

        #[link_name = "__FIREFLY_BC"]
        static BYTECODE_DATA: u8;
    }

    let bytes = unsafe { slice::from_raw_parts(&BYTECODE_DATA, BYTECODE_LEN) };

    let reader = BytecodeReader::new(bytes);
    reader.read().map(|code| Arc::new(code))
}
