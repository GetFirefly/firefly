use core::sync::atomic::{AtomicU8, Ordering};

use smallvec::SmallVec;

use firefly_system::time::SystemTime;

use crate::gc::Gc;
use crate::process::signals::{Message, Signal, SignalEntry};
use crate::term::{
    atoms, Atom, BigInt, Cons, Int, LayoutBuilder, Map, OpaqueTerm, Pid, Term, TermFragment,
    ToTerm, Tuple,
};

use super::registry::WeakAddress;
use super::system::{self, SystemMessage};

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum LogLevel {
    Info = 0,
    Warning,
    Error,
}
impl LogLevel {
    const fn pack(self) -> u8 {
        self as u8
    }

    #[inline]
    const fn unpack(raw: u8) -> Self {
        match raw {
            0 => Self::Info,
            1 => Self::Warning,
            2 => Self::Error,
            _ => unreachable!(),
        }
    }
}
impl Into<Term> for LogLevel {
    fn into(self) -> Term {
        Term::Atom(self.into())
    }
}
impl Into<OpaqueTerm> for LogLevel {
    fn into(self) -> OpaqueTerm {
        let atom: Atom = self.into();
        atom.into()
    }
}
impl Into<Atom> for LogLevel {
    fn into(self) -> Atom {
        match self {
            Self::Info => atoms::Info,
            Self::Warning => atoms::Warning,
            Self::Error => atoms::Error,
        }
    }
}

static WARNING_LEVEL: AtomicU8 = AtomicU8::new(LogLevel::Warning.pack());

/// Returns the [`LogLevel`] at which warnings will be logged
#[inline]
pub fn warning_level() -> LogLevel {
    LogLevel::unpack(WARNING_LEVEL.load(Ordering::Relaxed))
}

/// Sets the global [`LogLevel`] at which warnings will be logged
#[inline]
pub fn set_warning_level(level: LogLevel) {
    WARNING_LEVEL.store(level.pack(), Ordering::Relaxed);
}

/// Sends the given informational message to the logger
///
/// This will use the group leader of the error_logger process for output
#[inline]
pub fn info(message: &str) -> Result<(), ()> {
    send_info_to_logger(message, None)
}

/// Sends an informational message to the logger, using the provided format string and arguments term.
///
/// This will use the group leader of the error_logger process for output
#[inline]
pub fn format_info<T>(format: &str, args: T) -> Result<(), ()>
where
    T: ToTerm,
{
    send_term_to_logger(LogLevel::Info, None, format, args)
}

/// Sends the given warning message to the logger
///
/// This will use the group leader of the error_logger process for output
#[inline]
pub fn warning(message: &str) -> Result<(), ()> {
    send_warning_to_logger(message, None)
}

/// Sends a warning message to the logger, using the provided format string and arguments term.
///
/// This will use the group leader of the error_logger process for output
#[inline]
pub fn format_warning<T>(format: &str, args: T) -> Result<(), ()>
where
    T: ToTerm,
{
    send_term_to_logger(warning_level(), None, format, args)
}

/// Sends the given error message to the logger
///
/// This will use the group leader of the error_logger process for output
#[inline]
pub fn error(message: &str) -> Result<(), ()> {
    send_error_to_logger(message, None)
}

/// Sends an error message to the logger, using the provided format string and arguments term.
///
/// This will use the group leader of the error_logger process for output
#[inline]
pub fn format_error<T>(format: &str, args: T) -> Result<(), ()>
where
    T: ToTerm,
{
    send_term_to_logger(warning_level(), None, format, args)
}

/// Sends the given informational message to the logger
///
/// The group leader is optional, and controls where output will be logged to when provided.
pub fn send_info_to_logger(message: &str, group_leader: Option<Pid>) -> Result<(), ()> {
    send_to_logger(LogLevel::Info, group_leader, message)
}

/// Sends the given warning message to the logger
///
/// The actual log level at which the message will be sent is controlled by the global
/// warning level set by `set_warning_level`, which is `LogLevel::Warning` by default,
/// but can be set to other levels if desired.
///
/// The group leader is optional, and controls where output will be logged to when provided.
pub fn send_warning_to_logger(message: &str, group_leader: Option<Pid>) -> Result<(), ()> {
    send_to_logger(warning_level(), group_leader, message)
}

/// Sends the given error message to the logger
///
/// The group leader is optional, and controls where output will be logged to when provided.
pub fn send_error_to_logger(message: &str, group_leader: Option<Pid>) -> Result<(), ()> {
    send_to_logger(LogLevel::Error, group_leader, message)
}

/// Sends an error message to the logger with the given format string and argument(s) in term form
///
/// The group leader is optional, and controls where output will be logged to when provided.
pub fn send_error_term_to_logger<T>(
    format: &str,
    args: T,
    group_leader: Option<Pid>,
) -> Result<(), ()>
where
    T: ToTerm,
{
    send_term_to_logger(LogLevel::Error, group_leader, format, args)
}

#[inline]
fn send_to_logger(level: LogLevel, group_leader: Option<Pid>, message: &str) -> Result<(), ()> {
    send_logger_message(
        level,
        group_leader,
        "~s~n",
        Args::<Term>::Str(message),
        SystemTime::now(),
        Pid::current(),
    )
}

#[inline]
fn send_term_to_logger<T>(
    level: LogLevel,
    group_leader: Option<Pid>,
    format: &str,
    args: T,
) -> Result<(), ()>
where
    T: ToTerm,
{
    send_logger_message(
        level,
        group_leader,
        format,
        Args::Term(args),
        SystemTime::now(),
        Pid::current(),
    )
}

enum Args<'a, T> {
    Str(&'a str),
    Term(T),
}

/// Sends a message of the following format to the `error_logger` process:
///
/// ```erlang
///     {log, Level, Format, Args, #{gl => Gl, pid => Pid, time => Time, error_logger => #{Tag => Level, emulator => true}}}
/// ```
///
/// Where:
///
/// * `Level` is an atom representing the log level
/// * `Format` is a charlist representing the format string for the message
/// * `Args` is a list containing zero or more arguments to apply to the format string
/// * `Gl` is optional metadata, the group leader pid
/// * `Pid` is optional metadata, the pid of the logging process
/// * `Time` is the system time (in microseconds since the UNIX epoch) at which the logger message was created
/// * `Tag` is the log level mapped for backwards compatibility in error_logger
///
fn send_logger_message<T>(
    level: LogLevel,
    group_leader: Option<Pid>,
    format: &str,
    args: Args<'_, T>,
    time: SystemTime,
    pid: Option<Pid>,
) -> Result<(), ()>
where
    T: ToTerm,
{
    let time = time.duration_since(SystemTime::UNIX_EPOCH).unwrap();
    let time = time.as_micros() as i64;

    let mut layout = LayoutBuilder::new();
    layout.build_list(format.len());
    match &args {
        Args::Str(args) => {
            layout.build_list(args.len()).build_list(1);
        }
        Args::Term(t) => {
            layout += t.fragment_layout();
            layout.build_list(1);
        }
    }
    if time > Int::MAX_SMALL || time < Int::MIN_SMALL {
        layout.build_bigint();
    }
    let num_metadata = 2 + (group_leader.is_some() as usize) + (pid.is_some() as usize);
    if group_leader.is_some() {
        layout.build_pid();
    }
    if pid.is_some() {
        layout.build_pid();
    }
    layout.build_map(2).build_map(num_metadata).build_tuple(5);
    let fragment_ptr = layout.into_fragment().unwrap();
    let fragment = unsafe { fragment_ptr.as_ref() };

    let format = Cons::charlist_from_str(format, &fragment)
        .unwrap()
        .map(Term::Cons)
        .unwrap_or(Term::Nil);
    let args = match args {
        Args::Str(args) => Cons::charlist_from_str(args, &fragment)
            .unwrap()
            .map(Term::Cons)
            .unwrap_or(Term::Nil),
        Args::Term(t) => {
            let term = t.to_term(&fragment).unwrap();
            Cons::new_in(
                Cons {
                    head: term.into(),
                    tail: OpaqueTerm::NIL,
                },
                &fragment,
            )
            .map(Term::Cons)
            .unwrap()
        }
    };

    let time = if time > Int::MAX_SMALL || time < Int::MIN_SMALL {
        let mut boxed = Gc::<BigInt>::new_uninit_in(&fragment).unwrap();
        unsafe {
            boxed.write(BigInt::from(time));
            Term::BigInt(boxed.assume_init())
        }
    } else {
        Term::Int(time)
    };
    let gl = match group_leader {
        None => None,
        Some(gl) => Some(Term::Pid(Gc::<Pid>::new_in(gl, &fragment).unwrap())),
    };
    let pid_term = match pid {
        None => None,
        Some(ref pid) => Some(Term::Pid(
            Gc::<Pid>::new_in(pid.clone(), &fragment).unwrap(),
        )),
    };

    let tag = match level {
        LogLevel::Info => atoms::InfoMsg,
        LogLevel::Warning => atoms::WarningMsg,
        LogLevel::Error => atoms::Error,
    };

    let error_logger_metadata = unsafe {
        Map::from_sorted_slice(
            &[
                (atoms::Emulator.into(), atoms::True.into()),
                (atoms::Tag.into(), tag.into()),
            ],
            &fragment,
        )
        .unwrap()
    };

    let mut metadata = SmallVec::<[(OpaqueTerm, OpaqueTerm); 4]>::default();
    metadata.push((atoms::ErrorLogger.into(), error_logger_metadata.into()));
    if let Some(gl) = gl {
        metadata.push((atoms::Gl.into(), gl.into()));
    }
    if let Some(pid) = pid_term {
        metadata.push((atoms::Pid.into(), pid.into()))
    }
    metadata.push((atoms::Time.into(), time.into()));
    let metadata =
        unsafe { Term::Map(Map::from_sorted_slice(metadata.as_slice(), &fragment).unwrap()) };

    let elements = &[
        atoms::Log.into(),
        level.into(),
        format.into(),
        args.into(),
        metadata.into(),
    ];
    let message = Tuple::from_slice(elements, &fragment)
        .map(Term::Tuple)
        .unwrap();

    let sender = if let Some(pid) = pid {
        WeakAddress::Process(pid)
    } else {
        WeakAddress::System
    };
    let message = SignalEntry::new(Signal::Message(Message {
        sender,
        message: TermFragment {
            term: message.into(),
            fragment: Some(fragment_ptr),
        },
    }));

    system::send_system_message(SystemMessage::ErrorLogger { message });

    Ok(())
}
