use std::future::Future;
use std::ptr;
use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::{Arc, Weak};

use firefly_rt::process::{signals::SignalEntry, Process};
use firefly_rt::services::registry::{self, Registrant};
use firefly_rt::services::system::{self, SystemDispatcher, SystemMessage};
use firefly_rt::term::{atoms, OpaqueTerm, Term};

use tokio::sync::mpsc;

struct AsyncSystemDispatcher {
    sender: mpsc::UnboundedSender<SystemMessage>,
}
impl AsyncSystemDispatcher {
    fn new(sender: mpsc::UnboundedSender<SystemMessage>) -> Arc<Self> {
        Arc::new(Self { sender })
    }
}
impl SystemDispatcher for AsyncSystemDispatcher {
    fn set_system_logger(&self, logger: OpaqueTerm) -> OpaqueTerm {
        let logger: SystemLogger = logger.try_into().unwrap();
        logger.set().into()
    }

    #[inline]
    fn enqueue(&self, message: SystemMessage) {
        self.sender.send(message).ok();
    }
}

/// Starts the system dispatcher, registers it with the runtime, and begins processing messages in the background
pub fn start() -> impl Future<Output = ()> + Send + 'static {
    let (sender, receiver) = mpsc::unbounded_channel();

    let dispatcher = AsyncSystemDispatcher::new(sender);
    system::install_system_dispatcher(dispatcher as Arc<dyn SystemDispatcher>);

    run(receiver)
}

static SYSTEM_LOGGER: AtomicPtr<Process> = AtomicPtr::new(1usize as *mut Process);
const DISABLED: *mut Process = ptr::null_mut();
const DEFAULT: *mut Process = 1usize as *mut Process;

#[derive(Clone)]
#[repr(u8)]
enum SystemLogger {
    Disabled = 0,
    Default,
    Custom(Weak<Process>),
}
impl SystemLogger {
    fn get() -> Self {
        match SYSTEM_LOGGER.load(Ordering::Acquire) {
            DISABLED => Self::Disabled,
            DEFAULT => Self::Default,
            ptr => Self::Custom(unsafe { Weak::from_raw(ptr) }),
        }
    }

    fn set(self) -> Self {
        let ptr = match self {
            Self::Disabled => DISABLED,
            Self::Default => DEFAULT,
            Self::Custom(weak) => weak.into_raw().cast_mut(),
        };
        match SYSTEM_LOGGER.swap(ptr, Ordering::Release) {
            DISABLED => Self::Disabled,
            DEFAULT => Self::Default,
            ptr => Self::Custom(unsafe { Weak::from_raw(ptr) }),
        }
    }

    #[inline]
    fn is_disabled(&self) -> bool {
        match self {
            Self::Disabled => true,
            _ => false,
        }
    }

    fn try_resolve(&self) -> Option<Arc<Process>> {
        match self {
            Self::Disabled => None,
            Self::Default => match registry::get_by_name(atoms::Logger)? {
                Registrant::Process(p) => Some(p),
                _ => None,
            },
            Self::Custom(p) => p.upgrade(),
        }
    }
}
impl PartialEq for SystemLogger {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Disabled, Self::Disabled) => true,
            (Self::Default, Self::Default) => true,
            (Self::Custom(a), Self::Custom(b)) => Weak::ptr_eq(a, b),
            _ => false,
        }
    }
}
impl Into<OpaqueTerm> for SystemLogger {
    fn into(self) -> OpaqueTerm {
        match self {
            Self::Disabled => atoms::Undefined.into(),
            Self::Default => atoms::Logger.into(),
            Self::Custom(weak) => match weak.upgrade() {
                None => atoms::Logger.into(),
                Some(_) => OpaqueTerm::code(Weak::into_raw(weak) as usize),
            },
        }
    }
}
impl TryFrom<OpaqueTerm> for SystemLogger {
    type Error = ();

    fn try_from(term: OpaqueTerm) -> Result<Self, Self::Error> {
        match term.into() {
            Term::Atom(a) if a == atoms::Undefined => Ok(Self::Disabled),
            Term::Atom(a) if a == atoms::Logger => Ok(Self::Default),
            Term::Pid(pid) => match registry::get_by_pid(pid.as_ref()) {
                None => Ok(Self::Default),
                Some(ref process) => Ok(Self::Custom(Arc::downgrade(process))),
            },
            Term::Code(code) => Ok(Self::Custom(unsafe {
                Weak::from_raw(code as *const Process)
            })),
            _ => Err(()),
        }
    }
}

struct ResolvedSystemLogger {
    logger: SystemLogger,
    cache: Option<Arc<Process>>,
}
impl Default for ResolvedSystemLogger {
    fn default() -> Self {
        Self::new(SystemLogger::get())
    }
}
impl ResolvedSystemLogger {
    fn new(logger: SystemLogger) -> Self {
        let mut this = Self {
            logger,
            cache: None,
        };
        this.load();
        this
    }

    fn send(&mut self, message: Box<SignalEntry>) {
        if self.logger.is_disabled() {
            return;
        }

        if self.cache.is_none() {
            self.load();
        }

        if let Some(process) = self.cache.clone() {
            if let Ok(_) = process.send_signal(message) {
                return;
            } else {
                // The configured logger failed, reset to default
                self.logger = SystemLogger::Default;
                self.cache = None;
            }
        }
    }

    /// Detect any changes to the globally configured system logger
    fn maybe_reload(&mut self) {
        let logger = SystemLogger::get();
        if self.logger == logger {
            return;
        }
        self.load();
    }

    /// Try to resolve the system logger to a concrete Arc<Process> reference
    fn load(&mut self) {
        if self.logger.is_disabled() {
            return;
        }
        let cache = self.logger.try_resolve();
        // Update the cache if successful
        if let Some(cache) = cache {
            self.cache = Some(cache);
            return;
        }
        // If the default logger was selected and failed, empty the cache
        // We'll try again at a later time, perhaps the logger will be running then
        if let SystemLogger::Default = self.logger {
            self.cache = None;
            return;
        }
        // At this point the logger was either a custom logger, or resolved to
        // a specific address, but it is no longer alive. We must reset the logger
        // to its default setting and try to look up the 'kernel' logger
        self.logger = SystemLogger::Default;
        self.cache = self.logger.try_resolve();
    }
}

/// Runs the core system dispatcher loop, processing messages in the system message queue
async fn run(mut receiver: mpsc::UnboundedReceiver<SystemMessage>) {
    let mut sys_logger = ResolvedSystemLogger::default();
    while let Some(message) = receiver.recv().await {
        match message {
            SystemMessage::ErrorLogger { message } => {
                sys_logger.send(message);
                sys_logger.maybe_reload();
            }
        }
    }
}
