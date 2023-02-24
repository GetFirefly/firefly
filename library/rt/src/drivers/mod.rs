use alloc::boxed::Box;
use alloc::sync::Arc;
use core::any::Any;
use core::ffi::c_void;
use core::mem::MaybeUninit;

use firefly_system::sync::OnceLock;

use crate::term::{Port, Reference, ReferenceId};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum DriverError {
    /// Returned when loading a driver is unsupported
    Unsupported,
    /// A general error, no error code or other data available
    Failed,
    /// Unable to load the driver library
    Loader,
    /// Driver initialization failed
    Init,
    /// Unable to initialize the driver due to a missing symbol
    MissingSymbol,
    /// The name associated with the driver doesn't match the library it was loaded from
    NameMismatch,
    /// The driver failed with the given error code
    Code(u32),
    /// The driver failed due to a bad argument
    Badarg,
}

bitflags::bitflags! {
    /// These flags are used to indicate driver capabilities and configuration to the runtime system.
    pub struct DriverFlags: u32 {
        /// Represents the default driver flags for all drivers (i.e no special features enabled)
        const DEFAULT = 0;
        /// The runtime system uses port-level locking on all ports executing this driver instead of driver-level locking. For more information, see `erl_driver`.
        const USE_PORT_LOCKING = 1 << 0;
        /// Marks that driver instances can handle being called in the `output` and/or `outputv` callbacks
        /// although a driver instance has marked itself as busy (see erl_driver:set_busy_port).
        ///
        /// This flag is required for built-in drivers.
        const SOFT_BUSY = 1 << 1;
        /// Disables busy port message queue functionality. For more information, see `erl_driver:erl_drv_busy_msgq_limits`.
        const NO_BUSY_MSGQ = 1 << 2;
        /// When this flag is specified, the linked-in driver must manually acknowledge that the port has been successfully
        /// started using `erl_driver:erl_drv_init_ack()`.
        ///
        /// This allows the implementor to make the `erlang:open_port` exit with `badarg` after some initial asynchronous initialization has been done.
        const USE_INIT_ACK = 1 << 3;
    }
}
impl Default for DriverFlags {
    #[inline]
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// An opaque pointer to a platform/operating-system specific event object.
///
/// On Unix systems, the object corresponds to that expected by the functions select/poll,
/// i.e. the event object must be a socket or pipe (or other object that select/poll can use).
///
/// On Windows, the object corresponds to that expected by the Win32 API function WaitForMultipleObjects.
/// This places other restrictions on the event object; see the Win32 SDK documentation.
pub type DriverEvent = *mut core::ffi::c_void;

/// Used to communicate an Erlang timestamp to/from a driver
#[derive(Copy, Clone)]
#[repr(C)]
pub struct DriverNow {
    mega: u32,
    seconds: u32,
    micro: u32,
}

/// Represents a relative timestamp whose value depends on `DriverTimeUnit`
pub type DriverTime = i64;

/// Represents the various time units supported in the driver API
#[derive(Default, Copy, Clone, PartialEq, Eq)]
pub enum DriverTimeUnit {
    #[default]
    Seconds,
    Milliseconds,
    Microseconds,
    Nanoseconds,
}

/// When a driver creates a monitor for a process, a `DriverMonitor` is filled in.
///
/// It is semantically opaque, but in practice is equivalent to a `ReferenceId` value, and the associated
/// driver API functions can be used to convert to/from a `Reference`.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct DriverMonitor(ReferenceId);
impl TryFrom<Reference> for DriverMonitor {
    type Error = ();

    fn try_from(reference: Reference) -> Result<Self, Self::Error> {
        if reference.is_local() {
            Ok(Self(reference.id()))
        } else {
            Err(())
        }
    }
}
impl Into<Reference> for DriverMonitor {
    #[inline]
    fn into(self) -> Reference {
        Reference::new(self.0)
    }
}

/// This trait represents drivers which can be loaded by the runtime to provide functionality via ports
///
/// This trait is intended to provide static/global metadata about the driver, and provide the ability to
/// start an instance of the driver for use by a port. These instances must implement the `Driver` trait,
/// which is the primary API for driver implementations.
///
/// A `LoadableDriver` must be `Sync`, as an instance of it will be registered globally and interacted with
/// from any number of threads when starting instances of the driver for use with ports.
pub trait LoadableDriver: Any + Send + Sync {
    /// Called directly after the driver has been loaded by erl_ddll:load_driver/2 (actually when the driver is added to the driver list).
    ///
    /// The driver is to return `Err` if the driver cannot initialize for some reason.
    fn init(&self) -> Result<(), DriverError>;

    /// The driver name. It must correspond to the atom used in erlang:open_port/2, and the name of the driver library file (without the extension).
    fn name(&self) -> &str;

    /// Returns the major and minor version of this driver as a tuple.
    fn version(&self) -> (u32, u32);

    /// Returns the set of flags for this driver.
    ///
    /// These flags are used to control how the runtime system interacts with this driver.
    fn flags(&self) -> DriverFlags;

    /// Called each time a new port is opened using this driver
    ///
    /// The start function receives the still-initializing port handle which should be stored
    /// in the driver state, so that various driver APIs can be called which require the port
    /// handle.
    ///
    /// This function must return a boxed instance of the driver state to be associated with `port`, as
    /// all other port driver callbacks must be implemented using only that state, and the global
    /// state stored in the port driver itself.
    ///
    /// # SAFETY
    ///
    /// This function cannot use `port` yet, but it is safe to call `Arc::assume_init` on it
    /// to convert it to a `Arc<Port>` for storage in the driver state. The reason we pass it
    /// as `Arc<MaybeUninit<Port>>` is to make it clear in the function signature that this handle
    /// is not yet initialized and is thus unsafe for use at this point.
    fn start(
        &self,
        port: Arc<MaybeUninit<Port>>,
        command: &str,
    ) -> Result<Box<dyn Driver>, DriverError>;
}

/// This trait is implemented by instances of a `LoadableDriver` bound to a specific port
///
/// Typically, each port gets a unique instance of the implementing type, but that is not a strict requirement.
///
/// Implementors of this trait must be `Sync`, as some functions may be called from other threads.
pub trait Driver: Any + Send + Sync {
    /// Called when a port based on this driver is closed.
    fn stop(&self);

    /// Called when an Erlang process has sent data to the port with `Port ! {self(), {command, Data}}` or `erlang:port_command/2`.
    fn output(&self, buffer: &[u8]);

    /// Called when a driver event (specified in parameter event) is signaled. This is used to help asynchronous drivers "wake up" when something occurs.
    ///
    /// On Unix the event is a pipe or socket handle (or something that the select system call understands).
    ///
    /// On Windows the event is an Event or Semaphore (or something that the WaitForMultipleObjects API function understands).
    /// (Some trickery in the emulator allows more than the built-in limit of 64 Events to be used.)
    ///
    /// To use this with threads and asynchronous routines, create a pipe on Unix and an Event on Windows.
    /// When the routine completes, write to the pipe (use SetEvent on Windows), this makes the emulator call ready_input or ready_output.
    ///
    /// False events can occur. That is, calls to ready_input or ready_output although no real events are signaled.
    /// In reality, it is rare (and OS-dependant), but a robust driver must nevertheless be able to handle such cases.
    fn ready_input(&self, event: *mut ());

    /// See the docs for `ready_input`.
    fn ready_output(&self, event: *mut ());

    /// A special routine invoked with erlang:port_control/3. It works a little like an "ioctl" for Erlang drivers.
    /// The data specified to port_control/3 arrives in buf and len. The driver can send data back, using *rbuf and rlen.
    ///
    /// This is the fastest way of calling a driver and get a response. It makes no context switch in the Erlang emulator
    /// and requires no message passing. It is suitable for calling C function to get faster execution, when Erlang is too slow.
    ///
    /// If the driver wants to return data, it is to return it in rbuf. When control is called, *rbuf points to a default buffer
    /// of rlen bytes, which can be used to return data. Data is returned differently depending on the port control flags
    /// (those that are set with erl_driver:set_port_control_flags).
    ///
    /// If the flag is set to PORT_CONTROL_FLAG_BINARY, a binary is returned. Small binaries can be returned by writing the raw data
    /// into the default buffer. A binary can also be returned by setting *rbuf to point to a binary allocated with
    /// erl_driver:driver_alloc_binary. This binary is freed automatically after control has returned.
    /// The driver can retain the binary for read only access with erl_driver:driver_binary_inc_refc to be freed later with
    /// erl_driver:driver_free_binary. It is never allowed to change the binary after control has returned.
    /// If *rbuf is set to NULL, an empty list is returned.
    ///
    /// If the flag is set to 0, data is returned as a list of integers.
    /// Either use the default buffer or set *rbuf to point to a larger buffer allocated with erl_driver:driver_alloc.
    /// The buffer is freed automatically after control has returned.
    ///
    /// Using binaries is faster if more than a few bytes are returned.
    ///
    /// The return value is the number of bytes returned in *rbuf.
    fn control(&self, command: u32, buf: &[u8], rbuf: *mut *mut u8, rlen: usize) -> usize;

    /// Called any time after the driver's timer reaches 0. The timer is activated with erl_driver:driver_set_timer.
    /// No priorities or ordering exist among drivers, so if several drivers time out at the same time, anyone of them is called first.
    fn timeout(&self);

    /// Called whenever the port is written to.
    ///
    /// The port is to be in binary mode, see erlang:open_port/2.
    ///
    /// When the `std` feature is enabled, this uses the `std::io::IoSlice` type, otherwise a simple
    /// slice-of-slices is used, which is semantically equivalent, but lacks the conveniences provided
    /// by the former.
    #[cfg(feature = "std")]
    fn outputv<'a>(&self, data: std::io::IoSlice<'a>);

    #[cfg(not(feature = "std"))]
    fn outputv<'a>(&self, data: &[&'a [u8]]);

    /// Called after an asynchronous call has completed. The asynchronous call is started with erl_driver:driver_async.
    ///
    /// This function is called from the Erlang emulator thread, as opposed to the asynchronous function, which is called in some thread (if multi-threading is enabled).
    ///
    /// The `async_data` pointer is the same as provided to `driver_async`
    fn ready_async(&self, async_data: *mut c_void);

    /// Called when the port is about to be closed, and there is data in the driver queue that must be flushed before 'stop' can be called.
    fn flush(&self);

    /// Called from erlang:port_call/3. It works a lot like the control callback, but uses the external term format for input and output.
    ///
    /// `command` is an integer, obtained from the call from Erlang (the second argument to erlang:port_call/3).
    ///
    /// `buf` and `len` provide the arguments to the call (the third argument to erlang:port_call/3). They can be decoded using ei functions.
    ///
    /// `rbuf` points to a return buffer, `rlen` bytes long. The return data is to be a valid Erlang term in the external (binary) format.
    /// This is converted to an Erlang term and returned by erlang:port_call/3 to the caller.
    /// If more space than `rlen` bytes is needed to return data, `rbuf` can be set to memory allocated with erl_driver:driver_alloc.
    /// This memory is freed automatically after call has returned.
    ///
    /// The `Ok` return value is the number of bytes returned in `rbuf`.
    ///
    /// If any `Err` value is returned, `erlang:port_call/3` throws a badarg error.
    fn call(
        &self,
        command: u32,
        buf: &[u8],
        rbuf: *mut *mut u8,
        rlen: usize,
        flags: *mut u32,
    ) -> Result<usize, DriverError>;

    /// Called when a monitored process exits.
    ///
    /// The `state` argument is the state associated with the port for which the process is monitored (using `erl_driver:driver_monitor_process`)
    /// and the monitor corresponds to the `DriverMonitor` structure filled in when creating the monitor.
    ///
    /// The driver interface function `erl_driver:driver_get_monitored_process` can be used to retrieve the pid of the exiting process as an ErlDrvTermData.
    fn process_exit(&self, monitor: DriverMonitor);

    /// Called on behalf of `erl_driver:driver_select` when it is safe to close an event object.
    ///
    /// A typical implementation on Unix is to do close((int)event).
    ///
    /// Argument reserved is intended for future use and is to be ignored.
    ///
    /// In contrast to most of the other callback functions, `stop_select` is called independent of any port.
    /// No state argument is passed to the function. No driver lock or port lock is guaranteed to be held.
    /// The port that called `driver_select` can even be closed at the time `stop_select` is called.
    /// But it can also be the case that `stop_select` is called directly by `erl_driver:driver_select`.
    ///
    /// # SAFETY
    ///
    /// It is not allowed to call any functions in the driver API from `stop_select`.
    ///
    /// This strict limitation is because the volatile context in which `stop_select` can be called.
    fn stop_select(&self, event: DriverEvent, reserved: *mut ());
}

pub type DriverInitFn = unsafe extern "C" fn() -> Result<Box<dyn LoadableDriver>, DriverError>;

#[cfg(feature = "std")]
mod loader {
    use alloc::string::{String, ToString};
    use core::ops::Deref;
    use std::path::Path;

    use firefly_system::sync::RwLock;
    use libloading::Library;
    use rustc_hash::FxHasher;
    use smallvec::SmallVec;

    type HashMap<K, V> = hashbrown::HashMap<K, V, core::hash::BuildHasherDefault<FxHasher>>;

    use super::*;

    #[derive(Default)]
    pub struct DriverManager(RwLock<DriverManagerImpl>);

    impl DriverManager {
        pub fn load<S: AsRef<Path>>(&self, path: S, name: &str) -> Result<(), DriverError> {
            let mut path = path
                .as_ref()
                .canonicalize()
                .map_err(|_| DriverError::Loader)?;
            path.push(name);
            if cfg!(target_os = "macos") {
                path.set_extension("dylib");
            } else if cfg!(unix) {
                path.set_extension("so");
            } else if cfg!(windows) {
                path.set_extension("dll");
            }
            let path = path.into_boxed_path();

            let mut init_symbol = SmallVec::<[u8; 32]>::new();
            init_symbol.extend_from_slice(name.as_bytes());
            init_symbol.extend_from_slice(b"_init");

            let mut manager = self.0.write();
            manager.load(path, init_symbol.as_slice(), name)
        }

        pub fn get(&self, name: &str) -> Option<Arc<dyn LoadableDriver>> {
            let manager = self.0.read();
            manager.get(name)
        }
    }

    #[derive(Default)]
    struct DriverManagerImpl {
        loaded: HashMap<String, Arc<dyn LoadableDriver>>,
        libraries: HashMap<Box<Path>, Library>,
    }

    impl DriverManagerImpl {
        fn load(
            &mut self,
            path: Box<Path>,
            init_symbol: &[u8],
            name: &str,
        ) -> Result<(), DriverError> {
            use hashbrown::hash_map::Entry;

            let driver_init_fn = match self.libraries.entry(path) {
                Entry::Occupied(entry) => {
                    let lib = entry.get();
                    unsafe {
                        lib.get::<DriverInitFn>(init_symbol)
                            .map_err(|_| DriverError::MissingSymbol)?
                            .into_raw()
                    }
                }
                Entry::Vacant(entry) => {
                    let lib = unsafe {
                        Library::new(entry.key().deref()).map_err(|_| DriverError::Loader)?
                    };
                    let lib = entry.insert(lib);
                    unsafe {
                        lib.get::<DriverInitFn>(init_symbol)
                            .map_err(|_| DriverError::MissingSymbol)?
                            .into_raw()
                    }
                }
            };

            match self.loaded.entry(name.to_string()) {
                Entry::Occupied(_) => Ok(()),
                Entry::Vacant(entry) => {
                    let driver = unsafe { driver_init_fn()? };
                    if driver.name() != name {
                        Err(DriverError::NameMismatch)
                    } else {
                        driver.init().unwrap();
                        entry.insert(driver.into());
                        Ok(())
                    }
                }
            }
        }

        #[inline]
        fn get(&self, name: &str) -> Option<Arc<dyn LoadableDriver>> {
            self.loaded.get(name).cloned()
        }
    }
}

#[cfg(not(feature = "std"))]
mod loader {
    use super::*;

    #[derive(Default)]
    pub struct DriverManager;

    impl DriverManager {
        pub fn load(&self, _path: &str, _name: &str) -> Result<(), DriverError> {
            Err(DriverError::Unsupported)
        }

        pub fn get(&self, _name: &str) -> Option<Arc<dyn LoadableDriver>> {
            None
        }
    }
}

use self::loader::DriverManager;

static DRIVERS: OnceLock<DriverManager> = OnceLock::new();

/// Loads a driver `name` from `path`
///
/// It is expected that `path` is a directory containing a shared library whose base name is `name` (i.e. ignoring the extension).
/// It is expected that the library, when loaded, contains a symbol `<name>_init`, where `<name>` is the same value as `name`.
///
/// Upon loading the library, and once the driver init function symbol has been located, it will be called to produce an instance of the driver.
/// If this is successful, the `init` function of the driver instance will be called as well.
///
/// If an error occurs at any point during the above process, the driver is not considered loaded.
#[cfg(feature = "std")]
pub fn load<S: AsRef<std::path::Path>>(path: S, name: &str) -> Result<(), DriverError> {
    with_drivers(|drivers| drivers.load(path, name))
}

#[cfg(not(feature = "std"))]
pub fn load(path: &str, name: &str) -> Result<(), DriverError> {
    with_drivers(|drivers| drivers.load(path, name))
}

/// Retreives an instance of a loaded driver named `name`
///
/// If the driver doesn't exist, or has been unloaded, this function will return `None`.
pub fn get(name: &str) -> Option<Arc<dyn LoadableDriver>> {
    with_drivers(|drivers| drivers.get(name))
}

#[inline]
fn with_drivers<F, T>(callback: F) -> T
where
    F: FnOnce(&DriverManager) -> T,
{
    let drivers = DRIVERS.get_or_init(DriverManager::default);
    callback(drivers)
}
